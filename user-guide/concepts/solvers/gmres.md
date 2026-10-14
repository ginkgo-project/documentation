# GMRES

`gko::solver::Gmres<ValueType>` is the safest default Krylov method for **general
non-symmetric** linear systems. It builds a Krylov subspace one vector at a time
via the Arnoldi process, solves a least-squares problem in the projected subspace,
and recovers the iterate. To bound memory it restarts after at most
`krylov_dim` iterations, and can end a cycle earlier once the residual has
dropped by a given ratio (`restart_ratio`).

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
  - **Required.** Stopping criteria; see [Stopping criteria](../stopping-criteria.md).
* - `preconditioner`
  - `LinOpFactory`
  - Preconditioner factory; rebuilt on each `generate()`. Defaults to identity (no preconditioning).
* - `generated_preconditioner`
  - `LinOp`
  - Pre-built preconditioner that bypasses `preconditioner` when set. Defaults to `nullptr`.
* - `krylov_dim`
  - `size_type`
  - Restart length: the size of the Krylov subspace before a restart; controls memory and convergence. With `restart_ratio > 0` it caps the basis size instead (the memory bound). Defaults to `0`, which resolves to `gmres_default_krylov_dim = 100`.
* - `restart_ratio`
  - `remove_complex<ValueType>`
  - Ratio-based restart, in $[0, 1)$. A cycle ends once the residual has dropped to this fraction of its value at the start of the cycle; see [Ratio-based restart](#ratio-based-restart-restart_ratio). Defaults to `0`, which restarts every `krylov_dim` iterations.
* - `flexible`
  - `bool`
  - Activates [FGMRES](#flexible-gmres-fgmres) for variable preconditioners. Defaults to `false`.
* - `ortho_method`
  - `gmres::ortho_method`
  - Orthogonalisation strategy; see below. Defaults to `mgs`.
* - `nullspace`
  - `std::shared_ptr<const LinOp>`
  - Right nullspace $N(A)$ of a singular matrix, a `solver::Nullspace`. The solver removes it from the initial guess and from every preconditioned vector. See [Singular systems and nullspaces](singular-systems.md). Defaults to none.
* - `left_nullspace`
  - `std::shared_ptr<const LinOp>`
  - Left nullspace $N(A^H)$, a `solver::Nullspace`. Projects $b$ onto $\mathrm{range}(A)$; independent of `nullspace`, so set it together with `nullspace` whenever $b$ may have a component outside $\mathrm{range}(A)$. Defaults to none.
```

## Restart length (`krylov_dim`)

GMRES's per-iteration work and memory both scale with the size of the Krylov
subspace built so far: at iteration $k$ the algorithm performs $k$ inner products,
$k$ AXPYs, and stores $k+1$ basis vectors plus the upper-Hessenberg matrix. To
bound the cost, GMRES restarts after `krylov_dim` iterations: the current
approximation $x_k$ becomes the new initial guess and a fresh Krylov subspace is
built from the new residual.

Tuning `krylov_dim` is the main GMRES knob:

- **Small `krylov_dim`** (≤ 30): low memory, low per-iteration work, but
  potentially slow convergence; restarts discard information about the operator
  spectrum.
- **Large `krylov_dim`** (≥ 100): closer to true GMRES; faster convergence per
  outer iteration, but memory and orthogonalisation cost grow.
- **Default** (`gmres_default_krylov_dim = 100`): a reasonable middle ground for
  problems that fit comfortably in memory.

If the system is very ill-conditioned and convergence stalls, enlarge
`krylov_dim` rather than lengthening the iteration budget.

### Ratio-based restart (`restart_ratio`)

Setting `restart_ratio` to $\eta \in (0, 1)$ ends a cycle as soon as
$\|r_i - A d_i\| \le \eta \|r_i\|$, where $r_i$ is the residual at the start of
the cycle and $d_i$ the correction built so far. With several right-hand sides
the condition has to hold for every right-hand side that is still running. The
value must lie in $[0, 1)$; zero disables the criterion.

`restart_ratio` changes the meaning of `krylov_dim`. With a positive ratio,
`krylov_dim` is the maximum basis size of a cycle and the memory bound, and a
cycle ends at whichever of the two conditions is met first. The ratio is a
per-cycle reduction, not a convergence tolerance, so useful values are far
larger than a stopping tolerance: 1e-3 to 1e-1 when a cycle gains orders of
magnitude, and about 0.9 when it gains only a few percent. It has to be larger
than the reduction a full `krylov_dim` cycle achieves. Shorter cycles mean less
orthogonalisation work per cycle, at the cost of more restarts.

```cpp
auto solver_factory = gko::solver::Gmres<double>::build()
                          .with_krylov_dim(200u)       // memory cap
                          .with_restart_ratio(1e-2)    // end a cycle after a 100x gain
                          .with_criteria(gko::stop::max_iters(5000u),
                                         gko::stop::relative_residual_norm(1e-10))
                          .on(exec);
```

The criterion bounds the backward error of each cycle's correction, which lets
restarted GMRES be analysed like GMRES-based iterative refinement (see
Buttari et al. under Publications).

## Orthogonalisation strategies

The Arnoldi step orthogonalises the next basis vector against all previous ones.
The numerical-linear-algebra literature offers three standard recipes, all
exposed as the `ortho_method` factory parameter (`gko::solver::gmres::ortho_method`):

| Mode | Algorithm | Cost per Arnoldi step | Numerical stability | Collective / kernel-launch structure |
|---|---|---|---|---|
| `mgs` *(default)* | Modified Gram-Schmidt | $k$ dot products + $k$ AXPYs | Best of the three for finite-precision arithmetic. | $k$ separate rounds in a critical chain: each round's dot product depends on the previous round's AXPY. Each round is itself a fully parallel reduction over $n$ elements; the chain just cannot be batched. |
| `cgs` | Classical Gram-Schmidt | $k$ dot products + $k$ AXPYs | Worst of the three; loss of orthogonality grows quickly when the basis vectors are near-collinear. | All $k$ dot products fuse into one batched reduction (one $V^T w$ gemv); the AXPYs likewise fuse into one batched update. |
| `cgs2` | Classical Gram-Schmidt with one re-orthogonalisation pass | $2k$ dot products + $2k$ AXPYs | Comparable to MGS in practice. | Two batched reductions back-to-back; same per-pass structure as CGS, twice. |

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
preconditioners that change between Krylov iterations, for example, an inner
Krylov solver run to a loose tolerance, or a multigrid cycle with adaptively
chosen smoothers. Standard GMRES would silently produce a corrupted Krylov
subspace in that case because it implicitly assumes $M^{-1}$ is the same operator
every time it is applied.

```cpp
auto fgmres = gko::solver::Gmres<double>::build()
                  .with_krylov_dim(50u)
                  .with_flexible(true)
                  .with_preconditioner(  // an inner solve with a loose limit
                      gko::solver::Cg<double>::build().with_criteria(
                          gko::stop::max_iters(5u)))
                  .with_criteria(gko::stop::max_iters(1000u),
                                 gko::stop::relative_residual_norm(1e-8))
                  .on(exec);
```

The same `Gmres` template implements both
modes, switched by the `flexible` flag. The trade-off is memory: FGMRES has to
store $M^{-1} v_j$ for every Krylov basis vector (not just $v_j$), so the
basis-storage footprint roughly doubles compared to standard GMRES at the same
`krylov_dim`. The orthogonalisation cost is unchanged; the second basis still
has $k$ vectors per Arnoldi iteration.

## Right preconditioning

Ginkgo's GMRES applies the preconditioner on the right: the Arnoldi step builds
$K_m(A M^{-1}, r_0)$ rather than $K_m(M^{-1} A, M^{-1} r_0)$, and the solution
recovery on restart applies $M^{-1}$ to the projected coefficients. The
practical consequence is that the convergence test sees the **unpreconditioned**
residual $r = b - A x$, not $M^{-1} r$. See [Preconditioners: left, right, and
split](../preconditioners.md#left-right-and-split-preconditioning) for the
mathematical detail and the contrast with CG/BiCGSTAB.

## Memory footprint

Per right-hand side, GMRES stores the Krylov basis (`krylov_dim + 1` vectors of
length $n$), the upper-Hessenberg matrix (`(krylov_dim + 1) × krylov_dim`
scalars), the Givens-rotation arrays (`2 * krylov_dim` scalars), and a handful of
auxiliary vectors. With `krylov_dim = 100` this is about 100 full-length vectors
of overhead: substantial, and the dominant memory cost on top of the system
matrix and preconditioner. FGMRES roughly doubles the basis storage.

For very large systems where this overhead is prohibitive, switch to BiCGSTAB or
IDR(s): both have bounded per-iteration memory.

## Publications

- [A Modular Framework for the Backward Error Analysis of GMRES](https://doi.org/10.1093/imanum/draf049) (Buttari et al.), the analysis behind `restart_ratio`
- [Compressed Basis GMRES on High-Performance Graphics Processing Units](https://doi.org/10.1177/10943420221115140)
- [FRSZ2 for In-Register Block Compression Inside GMRES on GPUs](https://doi.org/10.1109/SCW63240.2024.00038)

:::{seealso}
- [Solver taxonomy](../solvers.md): where GMRES fits among the families.
- [BiCGSTAB](bicgstab.md), [IDR(s)](idr.md): bounded-memory alternatives for
  non-symmetric systems.
- [Preconditioner taxonomy](../preconditioners.md): what to pair with GMRES,
  including the right-preconditioning placement.
- [Stopping criteria](../stopping-criteria.md): required configuration.
- API reference: [`gko::solver::Gmres`](../../../api-reference/solver/gmres.md)
:::
