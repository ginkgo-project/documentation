# IC family (IC, ParIC, ParICT)

The Incomplete Cholesky family preconditions a **symmetric
positive-definite** system $A$ by computing an approximation $L L^H \approx
A$ and using its action as the preconditioner: at apply time, the
preconditioner solves $L L^H z = r$ by chaining a forward triangular solve
with its conjugate-transpose backward solve.

Three pieces work together:

- [`gko::factorization::Ic`](../../../api-reference/factorization/ic.md): exact
  IC(0), incomplete Cholesky with the input matrix's own sparsity pattern; its
  `algorithm` parameter selects the `sparselib` or `syncfree` implementation.
- [`gko::factorization::ParIc`](../../../api-reference/factorization/par-ic.md):
  fine-grained parallel IC(0); asynchronous fixed-point iteration of
  Chow & Patel.
- [`gko::factorization::ParIct`](../../../api-reference/factorization/par-ict.md):
  threshold-based IC; refines the sparsity pattern of $L$ iteratively
  alongside the numerical values.

The matching preconditioner LinOp,
[`gko::preconditioner::Ic`](../../../api-reference/preconditioner/ic.md), wraps
the factorisation result and configures the triangular-solve LinOp used at
`apply` time. The default solver is a direct
[`LowerTrs`](../solvers/direct.md); a Krylov inner solver can be
substituted for cases where the triangular solve is the bottleneck.

## When to use IC

- **SPD matrices** where Jacobi is too weak but a full factorisation would
  not fit in memory. IC's storage is bounded by the sparsity pattern of
  $A$ (for IC(0)) or by a tunable budget (for ParICT).
- **As the preconditioner inside [CG](../solvers/cg.md), [FCG](../solvers/fcg.md),
  or [Minres](../solvers.md)**: IC preserves symmetry, so the
  preconditioned operator stays SPD.
- **Pair with a fill-reducing reordering** ([AMD](../reordering/amd.md),
  [nested dissection](../reordering/nested-dissection.md)) for higher
  quality on mesh-based problems. The reordering improves the sparsity
  pattern that IC(0) is constrained to.

For non-symmetric matrices, use the [ILU family](ilu.md) instead.

## Picking a variant

| Variant | Algorithm | Best when |
|---|---|---|
| `factorization::Ic` | Exact IC(0); `algorithm` selects `sparselib` (default) or `syncfree` | Small to moderate matrices; the canonical correctness baseline. |
| `factorization::ParIc` | Asynchronous fixed-point IC(0) | Larger matrices on GPU or many-core CPU; same sparsity pattern as `Ic`. |
| `factorization::ParIct` | Threshold-based IC with sparsity-pattern refinement | When IC(0) is too sparse to be effective; ParICT improves the pattern at the cost of more storage and longer generation. |

## Construction

```cpp
// Standard IC(0) preconditioner inside CG
auto cg = gko::solver::Cg<double>::build()
              .with_criteria(
                  gko::stop::Iteration::build().with_max_iters(100u),
                  gko::stop::ResidualNorm<double>::build()
                      .with_reduction_factor(1e-8))
              .with_preconditioner(
                  gko::preconditioner::Ic<>::build()
                      .with_factorization(
                          gko::factorization::ParIc<double, gko::int32>::build()
                              .on(exec))
                      .on(exec))
              .on(exec);

// With an iterative inner solve for the triangular system
auto preconditioner = gko::preconditioner::Ic<gko::solver::Bicgstab<double>>::build()
    .with_factorization(
        gko::factorization::ParIc<double, gko::int32>::build().on(exec))
    .with_l_solver(
        gko::solver::Bicgstab<double>::build()
            .with_criteria(
                gko::stop::Iteration::build()
                    .with_max_iters(3u).on(exec))
            .on(exec))
    .on(exec);
```

The first template parameter of `preconditioner::Ic` is the triangular
solver type; the default is `solver::LowerTrs`. The $L^H$ solve is
constructed automatically as the conjugate-transpose of the $L$ solve, so
only one solver factory is configured.

## Factory parameters

| Parameter | Type | Purpose |
|---|---|---|
| `factorization` | `LinOpFactory` | The factorisation that produces the $L$ (and implicitly $L^H$) factor. Typically `Ic`, `ParIc`, or `ParIct`. Defaults to `ParIc`. |
| `l_solver` | `LinOpFactory` | Factory for the $L$ triangular solver. Defaults to a direct triangular solve. |

## Implementation notes

The triangular-solve dispatcher caches the lower factor and its
conjugate-transpose adapter, so each `apply` runs two triangular solves
without rebuilding the operator chain. ParIC and ParICT generation are
asynchronous fixed-point iterations of Chow & Patel; the number of
sweeps needed for a usable preconditioner depends on the parallelism
level and the problem (a single sweep is enough for sequential
execution; 3 sweeps on OpenMP and 10+ on GPU are typical).

:::{seealso}
- [ILU family](ilu.md): the same approach for non-symmetric matrices.
- [Solver taxonomy](../solvers.md): Krylov methods that pair naturally with IC.
- [Reordering and permutations](../reordering.md): AMD or ND to improve the IC(0) sparsity pattern.
- API reference: [`gko::preconditioner::Ic`](../../../api-reference/preconditioner/ic.md).

References:
- Chow, E., Patel, A. *Fine-Grained Parallel Incomplete LU Factorization.*
  SIAM Journal on Scientific Computing, 37 (2), C169–C193, 2015.
  <https://doi.org/10.1137/140968896>
- Anzt, H., Chow, E., Dongarra, J. *ParILUT: A Parallel Threshold ILU
  for GPUs.* IEEE IPDPS 2019, pp. 231–241.
  <https://doi.org/10.1109/IPDPS.2019.00033>
:::
