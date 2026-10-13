# `gko::batch`

Batched linear algebra: solve many small independent systems at once
on the same executor, sharing kernel launches and data movement to
keep accelerators saturated. Every type in this namespace has the
uniform-batch constraint — all items share the same dimensions and
the same sparsity pattern (CSR/ELL store the column indices and row
pointers **once**, common across the whole batch).

For background on when batched solvers help, see the
[Batched concept page](../../user-guide/concepts/batched.md).

## Right-hand sides and matrices

- [`MultiVector`](multi-vector.md) — the right-hand-side container:
  one dense block per batch item, all sharing the same shape.
- [`matrix::Csr`](matrix-csr.md) — batched compressed sparse row.
- [`matrix::Dense`](matrix-dense.md) — batched row-major dense.
- [`matrix::Ell`](matrix-ell.md) — batched equal-length-rows.
- [`matrix::Identity`](matrix-identity.md) — batched identity LinOp.

## Solvers

- [`solver::Cg`](solver-cg.md) — batched conjugate gradient (SPD).
- [`solver::Bicgstab`](solver-bicgstab.md) — batched stabilised
  biconjugate gradient (non-symmetric).
- [`solver::BatchSolver`](solver-batch-solver.md) — common runtime
  controls (tolerance, tolerance type, max-iterations) shared by every
  batched solver.

## Preconditioners

- [`preconditioner::Jacobi`](preconditioner-jacobi.md) — batched
  block-Jacobi.

## Loggers

- [`log::BatchConvergence`](log-batch-convergence.md) — per-batch-item
  iteration count and residual norm; the batched equivalent of
  `gko::log::Convergence`.

```{toctree}
:hidden:
:maxdepth: 1

multi-vector
matrix-csr
matrix-dense
matrix-ell
matrix-identity
solver-cg
solver-bicgstab
solver-batch-solver
preconditioner-jacobi
log-batch-convergence
```
