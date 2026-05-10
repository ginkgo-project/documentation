# Integrate Ginkgo with deal.II

:::{note}
Stub — content forthcoming.
:::

Use Ginkgo as the inner solver / preconditioner inside a deal.II
application. Covers translating between `dealii::SparseMatrix` and Ginkgo
`Csr` (full-copy and zero-copy paths), reusing deal.II's MPI communicator,
and integrating with deal.II's `SolverControl`.
