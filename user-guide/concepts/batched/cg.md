# Batched CG

`gko::batch::solver::Cg<ValueType>` is the batched conjugate-gradient solver. Each batch item is
solved with a short-recurrence CG iteration inside its own workgroup, and items run
independently — so a batch of $N$ symmetric positive-definite systems converges in parallel
without inter-item synchronisation.

## When to use

Batched CG is the right choice when **every** system in the batch is symmetric positive-definite
and the uniform-batch constraint
([same dimensions and sparsity pattern](../batched.md#the-uniform-batch-constraint))
holds. If even a few items in the batch are non-SPD, prefer
[Batched BiCGSTAB](bicgstab.md) — its convergence is more robust at the cost of two SpMVs and
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

| Parameter                          | Default                          | Effect                                                                 |
| ---------------------------------- | -------------------------------- | ---------------------------------------------------------------------- |
| `with_max_iterations(n)`           | (required)                       | Per-item iteration cap.                                                |
| `with_tolerance(tol)`              | (required)                       | Stopping threshold on the per-item residual.                           |
| `with_tolerance_type(t)`           | `tolerance_type::absolute`       | `absolute` checks $\lVert r_k \rVert$; `relative` checks $\lVert r_k \rVert / \lVert b \rVert$. |
| `with_preconditioner(factory)`     | `BatchIdentity` (no preconditioning) | Any `BatchLinOpFactory`. The preconditioner is generated per-item from the system matrix. |
| `with_generated_preconditioner(p)` | `nullptr`                        | Use a pre-built `BatchLinOp` instead of generating a new one. Mutually exclusive with `with_preconditioner`. |

The `tolerance_type` enum lives in `<ginkgo/core/stop/batch_stop_enum.hpp>` and accepts only
`absolute` or `relative`.

## Stopping rule and the implicit residual

CG checks convergence per-item against the **implicit** residual computed during the iteration
(the $\rho_k$ that drops out of the orthogonality recurrence) — *not* a freshly recomputed
$\lVert b - A x_k \rVert$. The implicit residual can drift from the true residual on
ill-conditioned problems; if your batch contains items with widely varying conditioning, run an
a-posteriori check on the returned solution and re-solve the items that did not converge to
your tolerance.

A single stopping configuration applies to all items in the batch — there is no per-item
tolerance setter. Each item, however, exits the inner loop independently: a converged item is
flagged as such and skipped on subsequent iterations while the others keep working.

## Runtime setters

The base class `BatchSolver` exposes `reset_tolerance(tol)`, `reset_max_iterations(n)`, and
`reset_tolerance_type(t)` for tuning the stopping rule between solves without rebuilding the
solver. `set_preconditioner_base(p)` swaps the preconditioner. These avoid paying the
factory-`generate` cost when you only want to tweak the stop conditions.

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
- [Batched solvers — overview](../batched.md) — the uniform-batch constraint and storage layout.
- [Batched BiCGSTAB](bicgstab.md) — the alternative for non-symmetric batches.
- [Batched convergence logger](convergence.md) — extracting per-item iteration counts and
  residuals.
- [Single-system CG](../solvers/cg.md) — the same algorithm without the batch dimension.
- API reference: [`gko::batch::solver::Cg`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1batch_1_1solver_1_1Cg.html)
:::
