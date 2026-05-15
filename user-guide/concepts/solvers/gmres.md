# GMRES

`gko::solver::Gmres<ValueType>` is the safest default Krylov method for **general
non-symmetric** linear systems. It builds a Krylov subspace one vector at a time
via the Arnoldi process, solves a least-squares problem in the projected subspace,
and recovers the iterate. To bound memory it restarts every `krylov_dim`
iterations.

## When to use GMRES

- **Non-symmetric or indefinite systems** → GMRES is the safe default. It is
  guaranteed not to break down (unlike BiCGSTAB on some matrices) and can
  converge for any non-singular system given enough iterations.
- **Memory-tight non-symmetric systems** → consider [BiCGSTAB](bicgstab.md) or
  [IDR(s)](idr.md) instead; both have bounded memory unlike GMRES, whose
  memory grows with `krylov_dim`.
- **Variable preconditioner** (e.g. an inner Krylov solve, an adaptive multigrid
  smoother) → enable [FGMRES](#flexible-gmres-fgmres) via `with_flexible(true)`
  on this same class.
- **SPD systems** → use [CG](cg.md) instead; GMRES would work, but CG's three-term
  recurrence is dramatically cheaper.

## Construction

```cpp
auto gmres = gko::solver::Gmres<double>::build()
                 .with_krylov_dim(50u)
                 .with_criteria(
                     gko::stop::Iteration::build()
                         .with_max_iters(1000u)
                         .on(exec),
                     gko::stop::ResidualNorm<double>::build()
                         .with_reduction_factor(1e-8)
                         .on(exec))
                 .with_preconditioner(
                     gko::preconditioner::Ilu<>::build().on(exec))
                 .on(exec);

auto solver = gmres->generate(system_matrix);
solver->apply(b, x);
```

## Factory parameters

```{list-table}
:widths: 28 25 47
:header-rows: 1

* - Parameter
  - Type
  - Purpose
* - `criteria`
  - vector of `stop::CriterionFactory`
  - **Required.** Stopping criteria — see [Stopping criteria](../stopping-criteria.md).
* - `preconditioner`
  - `LinOpFactory`
  - Preconditioner factory; rebuilt on each `generate()`. Defaults to identity (no preconditioning).
* - `generated_preconditioner`
  - `LinOp`
  - Pre-built preconditioner that bypasses `preconditioner` when set. Defaults to `nullptr`.
* - `krylov_dim`
  - `size_type`
  - Restart length — the size of the Krylov subspace before a restart; controls memory and convergence. Defaults to `0`, which resolves to `gmres_default_krylov_dim = 100`.
* - `flexible`
  - `bool`
  - Activates [FGMRES](#flexible-gmres-fgmres) for variable preconditioners. Defaults to `false`.
* - `ortho_method`
  - `gmres::ortho_method`
  - Orthogonalisation strategy — see below. Defaults to `mgs`.
```

## Restart length (`krylov_dim`)

GMRES's per-iteration work and memory both scale with the size of the Krylov
subspace built so far: at iteration $k$ the algorithm performs $k$ inner products,
$k$ AXPYs, and stores $k+1$ basis vectors plus the upper-Hessenberg matrix. To
bound the cost, GMRES restarts every `krylov_dim` iterations: the current
approximation $x_k$ becomes the new initial guess and a fresh Krylov subspace is
built from the new residual.

Tuning `krylov_dim` is the main GMRES knob:

- **Small `krylov_dim`** (≤ 30): low memory, low per-iteration work, but
  potentially slow convergence — restarts discard information about the operator
  spectrum.
- **Large `krylov_dim`** (≥ 100): closer to true GMRES; faster convergence per
  outer iteration, but memory and orthogonalisation cost grow.
- **Default** (`gmres_default_krylov_dim = 100`): a reasonable middle ground for
  problems that fit comfortably in memory.

If the system is very ill-conditioned and convergence stalls, enlarge
`krylov_dim` rather than lengthening the iteration budget.

## Orthogonalisation strategies

The Arnoldi step orthogonalises the next basis vector against all previous ones.
The numerical-linear-algebra literature offers three standard recipes, all
exposed as the `ortho_method` factory parameter (`gko::solver::gmres::ortho_method`):

| Mode | Algorithm | Cost per Arnoldi step | Numerical stability | Collective / kernel-launch structure |
|---|---|---|---|---|
| `mgs` *(default)* | Modified Gram-Schmidt | $k$ dot products + $k$ AXPYs | Best of the three for finite-precision arithmetic. | $k$ separate rounds in a critical chain — each round's dot product depends on the previous round's AXPY. Each round is itself a fully parallel reduction over $n$ elements; the chain just cannot be batched. |
| `cgs` | Classical Gram-Schmidt | $k$ dot products + $k$ AXPYs | Worst of the three; loss of orthogonality grows quickly when the basis vectors are near-collinear. | All $k$ dot products fuse into one batched reduction (one $V^T w$ gemv); the AXPYs likewise fuse into one batched update. |
| `cgs2` | Classical Gram-Schmidt with one re-orthogonalisation pass | $2k$ dot products + $2k$ AXPYs | Comparable to MGS in practice. | Two batched reductions back-to-back — same per-pass structure as CGS, twice. |

Selection guidance:

- **CPU and small-to-medium Krylov dim** → `mgs` is fine. Each round's reduction is
  large enough that kernel-launch / collective overhead is not the bottleneck.
- **GPU or distributed-memory** → prefer `cgs2`. The chain of $k$ small reductions
  in `mgs` becomes a bottleneck once the per-rank or per-warp work shrinks: each
  reduction pays a fixed kernel-launch or all-reduce latency, and they cannot be
  batched. `cgs2` costs roughly twice as many flops as `mgs` but collapses the
  inner products into one batched reduction per pass, which maps well to GPUs and
  to MPI all-reduces.
- **`cgs`** → only when you have explicit reason to believe the basis stays well
  conditioned (e.g., a small `krylov_dim` and a cheap preconditioner) and you can
  afford to lose stability for raw throughput. As a default choice it is rarely
  the right one; reach for `cgs2` instead.

## Flexible GMRES (FGMRES)

Setting `with_flexible(true)` activates the **flexible** variant. FGMRES tolerates
preconditioners that change between Krylov iterations — for example, an inner
Krylov solver run to a loose tolerance, or a multigrid cycle with adaptively
chosen smoothers. Standard GMRES would silently produce a corrupted Krylov
subspace in that case because it implicitly assumes $M^{-1}$ is the same operator
every time it is applied.

```cpp
auto fgmres = gko::solver::Gmres<double>::build()
                  .with_krylov_dim(50u)
                  .with_flexible(true)
                  .with_preconditioner(/* possibly varying inner solver */)
                  .with_criteria(/* ... */)
                  .on(exec);
```

There is no separate `Fgmres` class; the same `Gmres` template implements both
modes, switched by the `flexible` flag. The trade-off is memory: FGMRES has to
store $M^{-1} v_j$ for every Krylov basis vector (not just $v_j$), so the
basis-storage footprint roughly doubles compared to standard GMRES at the same
`krylov_dim`. The orthogonalisation cost is unchanged — the second basis still
has $k$ vectors per Arnoldi iteration.

## Right preconditioning

Ginkgo's GMRES applies the preconditioner on the right: the Arnoldi step builds
$K_m(A M^{-1}, r_0)$ rather than $K_m(M^{-1} A, M^{-1} r_0)$, and the solution
recovery on restart applies $M^{-1}$ to the projected coefficients. The
practical consequence is that the convergence test sees the **unpreconditioned**
residual $r = b - A x$, not $M^{-1} r$. See [Preconditioners — left, right, and
split](../preconditioners.md#left-right-and-split-preconditioning) for the
mathematical detail and the contrast with CG/BiCGSTAB.

## Memory footprint

Per right-hand side, GMRES stores the Krylov basis (`krylov_dim + 1` vectors of
length $n$), the upper-Hessenberg matrix (`(krylov_dim + 1) × krylov_dim`
scalars), the Givens-rotation arrays (`2 * krylov_dim` scalars), and a handful of
auxiliary vectors. With `krylov_dim = 100` this is about 100 full-length vectors
of overhead — substantial, and the dominant memory cost on top of the system
matrix and preconditioner. FGMRES roughly doubles the basis storage.

For very large systems where this overhead is prohibitive, switch to BiCGSTAB or
IDR(s) — both have bounded per-iteration memory.

## Publications

- [Compressed Basis GMRES on High-Performance Graphics Processing Units](https://doi.org/10.1177/10943420221115140)
- [FRSZ2 for In-Register Block Compression Inside GMRES on GPUs](https://doi.org/10.1109/SCW63240.2024.00038)

:::{seealso}
- [Solvers — taxonomy](../solvers.md) — where GMRES fits among the families.
- [BiCGSTAB](bicgstab.md), [IDR(s)](idr.md) — bounded-memory alternatives for
  non-symmetric systems.
- [Preconditioners — taxonomy](../preconditioners.md) — what to pair with GMRES,
  including the right-preconditioning placement.
- [Stopping criteria](../stopping-criteria.md) — required configuration.
- API reference: [`gko::solver::Gmres`](../../../api-reference/solver/gmres.md)
:::
