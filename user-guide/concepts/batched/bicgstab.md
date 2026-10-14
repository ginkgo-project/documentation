# Batched BiCGSTAB

`gko::batch::solver::Bicgstab<ValueType>` is the batched bi-conjugate-gradient stabilised
solver. Each batch item runs an independent BiCGSTAB iteration inside its own workgroup, so a
batch of $N$ general (possibly non-symmetric) systems converges in parallel without inter-item
synchronisation.

## When to use

Batched BiCGSTAB is the safe default for batches whose items are **general**: non-symmetric or
indefinite. It uses two SpMVs and two preconditioner applies per iteration (about twice the
work of CG), so prefer [Batched CG](cg.md) when every item in the batch is symmetric
positive-definite.

The uniform-batch constraint
([same dimensions and sparsity pattern](../batched.md#the-uniform-batch-constraint)) applies
the same way as for CG.

## Construction

```cpp
namespace b = gko::batch;

auto solver = b::solver::Bicgstab<double>::build()
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
| `with_max_iterations(n)` | `int` | Per-item iteration cap. Defaults to 100. |
| `with_tolerance(tol)` | `double` | Stopping threshold on the per-item residual. Defaults to `1e-11`. |
| `with_tolerance_type(t)` | `gko::batch::stop::tolerance_type` | `absolute` (default) checks $\lVert r_k \rVert$; `relative` checks $\lVert r_k \rVert / \lVert b \rVert$. |
| `with_preconditioner(factory)` | `std::shared_ptr<const BatchLinOpFactory>` | Preconditioner generated per item from the system matrix. Defaults to no preconditioner (a batched identity). |
| `with_generated_preconditioner(p)` | `std::shared_ptr<const BatchLinOp>` | A pre-built preconditioner. When it is set, `with_preconditioner` is ignored. Defaults to `nullptr`. |

All parameters are optional: a solver built with `.on(exec)` alone uses the defaults.

The `tolerance_type` enum lives in `<ginkgo/core/stop/batch_stop_enum.hpp>` and accepts only
`absolute` or `relative`.

The parameters match [Batched CG](cg.md#factory-parameters); only the algorithm differs.

## Stopping rule and breakdown

BiCGSTAB monitors the **implicit** residual (the same caveat as CG, see
[Stopping rule](cg.md#stopping-rule-and-the-implicit-residual)). The algorithm can also
break down when the inner $\omega$ scalar in the stabilisation step gets close to zero; in
practice this manifests as a per-item iteration that stalls without reaching the requested
tolerance. Run an a-posteriori check on the returned solution if your batch contains
near-singular or strongly non-normal items.

Each item in the batch exits the inner loop independently; a converged item is flagged and
skipped on subsequent iterations while the others continue.

## Per-iteration cost

BiCGSTAB performs **two** SpMVs, **two** preconditioner applies, **four** dot products, and a
handful of AXPY-style updates per iteration. The doubled SpMV / preconditioner cost compared
to CG is the price for handling non-symmetric systems with bounded (rather than growing)
memory.

The right preconditioning placement (consistent with the single-system Bicgstab) means each
preconditioner apply happens *before* the SpMV that builds the next Krylov direction.

## Runtime setters

`BatchSolver` exposes `reset_tolerance(tol)`, `reset_max_iterations(n)` and
`reset_tolerance_type(t)`, the same surface as Batched CG. To use a different preconditioner,
generate a new solver with `with_generated_preconditioner(p)` or `with_preconditioner(factory)`.

## Inspecting per-item convergence

Attach a [`BatchConvergence`](convergence.md) logger before `apply` to retrieve per-item
iteration counts and final residual norms after the solve.

:::{seealso}
- [Batched solvers](../batched.md): the uniform-batch constraint and storage layout.
- [Batched CG](cg.md): the cheaper alternative for SPD batches.
- [Batched convergence logger](convergence.md): extracting per-item iteration counts and
  residuals.
- [Single-system BiCGSTAB](../solvers/bicgstab.md): the same algorithm without the batch
  dimension.
- API reference: [`gko::batch::solver::Bicgstab`](../../../api-reference/batch/solver-bicgstab.md)
:::
