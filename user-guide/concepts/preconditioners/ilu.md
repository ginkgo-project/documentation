# ILU family (ILU, ParILU, ParILUT)

The Incomplete LU family preconditions a **general non-symmetric** system
$A$ by computing an approximation $L U \approx A$ and using its action as
the preconditioner — at apply time, the preconditioner solves $L U z = r$
by chaining a forward triangular solve with $L$ and a backward triangular
solve with $U$.

Three pieces work together:

- [`gko::factorization::Ilu`](../../../api-reference/factorization/ilu.md) — exact
  ILU(0): incomplete LU with the input matrix's own sparsity pattern.
- [`gko::factorization::ParIlu`](../../../api-reference/factorization/par-ilu.md) —
  fine-grained parallel ILU(0); asynchronous fixed-point iteration of
  Chow & Patel.
- [`gko::factorization::ParIlut`](../../../api-reference/factorization/par-ilut.md) —
  threshold-based ILU; refines the sparsity patterns of $L$ and $U$
  iteratively alongside the numerical values.

The matching preconditioner LinOp,
[`gko::preconditioner::Ilu`](../../../api-reference/preconditioner/ilu.md), wraps
the factorisation result and configures the two triangular-solve LinOps used
at `apply` time. Default solvers are direct
[`LowerTrs`](../solvers/direct.md) / `UpperTrs`; Krylov inner solvers can be
substituted for cases where the triangular solves are the bottleneck.

## When to use ILU

- **Non-symmetric matrices** where Jacobi is too weak but a full LU
  factorisation would not fit in memory. ILU's storage is bounded by the
  sparsity pattern of $A$ (for ILU(0)) or by a tunable budget (for ParILUT).
- **As the preconditioner inside [BiCGSTAB](../solvers/bicgstab.md),
  [GMRES](../solvers/gmres.md), or [IDR(s)](../solvers/idr.md)** — ILU
  does not preserve symmetry, so non-symmetric Krylov methods are the
  natural pairing.
- **Pair with a fill-reducing reordering** ([AMD](../reordering/amd.md),
  [nested dissection](../reordering/nested-dissection.md)) and/or an
  [MC64](../reordering/mc64.md) matching pre-step on indefinite systems
  to improve diagonal dominance before factorisation.

For SPD matrices, use the [IC family](ic.md) instead — it preserves
symmetry and lets you pair with CG.

## Picking a variant

| Variant | Algorithm | Best when |
|---|---|---|
| `factorization::Ilu` | Exact ILU(0) on the input sparsity pattern: cuSPARSE/hipSPARSE or Ginkgo's sync-free kernel, chosen by the `algorithm` parameter; runs on the Reference, OpenMP, CUDA and HIP executors | Small to moderate matrices; the canonical correctness baseline. |
| `factorization::ParIlu` | Asynchronous fixed-point ILU(0) | Larger matrices on GPU or many-core CPU; same sparsity pattern as `Ilu`. |
| `factorization::ParIlut` | Threshold-based ILU with sparsity-pattern refinement | When ILU(0) is too sparse — ParILUT improves the pattern at the cost of more storage and longer generation. |

## Construction

```cpp
// Standard ILU(0) preconditioner inside BiCGSTAB
auto bicgstab = gko::solver::Bicgstab<double>::build()
                    .with_criteria(gko::stop::max_iters(1000u),
                                   gko::stop::relative_residual_norm(1e-8))
                    .with_preconditioner(
                        gko::preconditioner::Ilu<>::build()
                            .with_factorization(
                                gko::factorization::ParIlu<double, gko::int32>
                                    ::build().on(exec))
                            .on(exec))
                    .on(exec);

// With iterative inner solves for the triangular systems
auto preconditioner =
    gko::preconditioner::Ilu<double, false, gko::int32>::build()
        .with_factorization(
            gko::factorization::ParIlu<double, gko::int32>::build())
        .with_l_solver(
            gko::solver::Bicgstab<double>::build().with_criteria(
                gko::stop::max_iters(3u)))
        .with_u_solver(
            gko::solver::Bicgstab<double>::build().with_criteria(
                gko::stop::max_iters(3u)))
        .on(exec);
```

`preconditioner::Ilu<ValueType, ReverseApply, IndexType>` defaults to
`<default_precision, false, int32>`. The $L$ and $U$ solvers are chosen at
runtime through `with_l_solver` / `with_u_solver`, which accept any
`LinOpFactory`; when unset, `Ilu` generates `solver::LowerTrs<ValueType,
IndexType>` and `solver::UpperTrs<ValueType, IndexType>`. With
`ReverseApply = true` the preconditioner solves with $U$ first, then $L$.
`get_l_solver()` and `get_u_solver()` return `std::shared_ptr<const LinOp>`;
use `gko::as<ConcreteType>(...)` to reach solver-specific accessors.

## Factory parameters

| Parameter | Type | Purpose |
|---|---|---|
| `factorization` | `LinOpFactory` | Factorisation that produces $L$ and $U$: `Ilu`, `ParIlu` or `ParIlut`. Defaults to `ParIlu` with default parameters. Not used when `generate()` receives a `Composition` holding $L$ and $U$ directly. |
| `l_solver` | `LinOpFactory` | Factory for the lower-triangular solver. Defaults to `solver::LowerTrs`. |
| `u_solver` | `LinOpFactory` | Factory for the upper-triangular solver. Defaults to `solver::UpperTrs`. |

## Implementation notes

ILU does **not** preserve symmetry of the system matrix, so the
preconditioned operator is generally non-symmetric even when $A$ is — use
non-symmetric Krylov methods (BiCGSTAB, GMRES, IDR(s)) with ILU rather
than CG.

ParILU and ParILUT generation are asynchronous fixed-point iterations of
Chow & Patel; the number of sweeps needed for a usable preconditioner
depends on the parallelism level (typically 3 on OpenMP, 10+ on GPU).
ParILUT additionally alternates *add-fill-in / fixed-point iterate /
drop-smallest* steps to refine the sparsity pattern.

:::{seealso}
- [IC family](ic.md) — the SPD-preserving counterpart of ILU.
- [Solvers — taxonomy](../solvers.md) — non-symmetric Krylov methods that
  pair naturally with ILU.
- [Reordering and permutations](../reordering.md) — AMD / ND for fill-in
  reduction; MC64 for diagonal stability.
- API reference: [`gko::preconditioner::Ilu`](../../../api-reference/preconditioner/ilu.md).

References:
- Chow, E., Patel, A. *Fine-Grained Parallel Incomplete LU Factorization.*
  SIAM Journal on Scientific Computing, 37 (2), C169–C193, 2015.
  <https://doi.org/10.1137/140968896>
- Anzt, H., Chow, E., Dongarra, J. *ParILUT — A Parallel Threshold ILU
  for GPUs.* IEEE IPDPS 2019, pp. 231–241.
  <https://doi.org/10.1109/IPDPS.2019.00033>
:::
