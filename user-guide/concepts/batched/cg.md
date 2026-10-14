# Batched CG

`gko::batch::solver::Cg<ValueType>` is the batched conjugate-gradient solver. Each batch item is
solved with a short-recurrence CG iteration inside its own workgroup, and items run
independently, so a batch of $N$ symmetric positive-definite systems converges in parallel
without inter-item synchronisation.

## When to use

Batched CG is the right choice when **every** system in the batch is symmetric positive-definite
and the uniform-batch constraint
([same dimensions and sparsity pattern](../batched.md#the-uniform-batch-constraint))
holds. If even a few items in the batch are non-SPD, prefer
[Batched BiCGSTAB](bicgstab.md); its convergence is more robust at the cost of two SpMVs and
two preconditioner applies per iteration.

## Construction

```cpp
namespace b = gko::batch;

auto solver = b::solver::Cg<double>::build()
                  .with_max_iterations(500)
                  .with_tolerance(1e-8)
                  .with_tolerance_type(b::stop::tolerance_type::relative)
                  // optional preconditioner factory:
                  .with_preconditioner(
                      b::preconditioner::Jacobi<double, int>::build().on(exec))
                  .on(exec)
                  ->generate(batched_matrix);

solver->apply(batched_rhs, batched_x);
```

## Factory parameters

| Parameter | Type | Purpose |
|---|---|---|
| `max_iterations` | `int` | Per-item iteration cap; default `100`. |
| `tolerance` | `double` | Stopping threshold on the per-item residual norm; default `1e-11`. |
| `tolerance_type` | `gko::batch::stop::tolerance_type` | `absolute` (default) checks $\lVert r_k \rVert$; `relative` checks $\lVert r_k \rVert / \lVert b \rVert$. |
| `preconditioner` | `std::shared_ptr<const BatchLinOpFactory>` | Preconditioner factory, generated from the batch matrix; the batch identity (`gko::batch::matrix::Identity`) when unset. |
| `generated_preconditioner` | `std::shared_ptr<const BatchLinOp>` | Pre-built preconditioner; when set it takes precedence over `preconditioner`. |

The `tolerance_type` enum lives in `<ginkgo/core/stop/batch_stop_enum.hpp>`.

## Stopping rule and the implicit residual

CG checks convergence per-item against the **implicit** residual computed during the iteration
(the $\rho_k$ that drops out of the orthogonality recurrence), *not* a freshly recomputed
$\lVert b - A x_k \rVert$. The implicit residual can drift from the true residual on
ill-conditioned problems; if your batch contains items with widely varying conditioning, run an
a-posteriori check on the returned solution and re-solve the items that did not converge to
your tolerance.

A single stopping configuration applies to all items in the batch. Each item, however, exits
the inner loop independently: a converged item is
flagged as such and skipped on subsequent iterations while the others keep working.

## Runtime setters

The base class `BatchSolver` exposes `reset_tolerance(tol)`, `reset_max_iterations(n)`, and
`reset_tolerance_type(t)` for tuning the stopping rule between solves without rebuilding the
solver, avoiding the factory-`generate` cost when you only want to tweak the stop
conditions. To use a different preconditioner, generate a new solver, passing a pre-built
preconditioner through `with_generated_preconditioner` to skip its setup.

## Per-iteration cost

CG performs one SpMV, one preconditioner apply, two dot products, and three AXPY-style updates
per iteration. Inside the fused kernel each of these is a workgroup-scoped operation on the
shared / global arrays for the item; reductions go through the cooperative-groups subgroup
described on the [batched overview](../batched.md#the-fused-kernel-design).

## Inspecting per-item convergence

Attach a [`BatchConvergence`](convergence.md) logger before calling `apply` to retrieve per-item
iteration counts and final residual norms after the solve. Without a logger the per-item
convergence information is discarded.

:::{seealso}
- [Batched solvers](../batched.md): the uniform-batch constraint and storage layout.
- [Batched BiCGSTAB](bicgstab.md): the alternative for non-symmetric batches.
- [Batched convergence logger](convergence.md): extracting per-item iteration counts and
  residuals.
- [Single-system CG](../solvers/cg.md): the same algorithm without the batch dimension.
- API reference: [`gko::batch::solver::Cg`](../../../api-reference/batch/solver-cg.md)
:::
