# Use a batched solver

:::{note}
Stub — content forthcoming.
:::

Pack many small independent systems into a `batch::matrix::Csr`, choose
between `batch::solver::Cg` and `batch::solver::Bicgstab`, attach a
`batch::log::BatchConvergence` logger, and read per-item iteration counts
and residuals back out. Covers the uniform-batch constraint and the shared
`col_idxs` / `row_ptrs` storage layout.
