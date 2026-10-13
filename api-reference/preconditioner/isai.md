# `gko::preconditioner::Isai`

Incomplete Sparse Approximate Inverse preconditioner. Computes a sparse
approximation $M_A \approx A^{-1}$ (or its triangular- / Cholesky-factor
variants) directly, so `apply` is a single sparse matrix-vector product
without any triangular solve. Useful when triangular substitution
becomes a serial bottleneck on highly-parallel hardware.

Rows with more than 32 non-zeros are handled through an excess system
(`excess_limit`, `excess_solver_factory`), so every entry of the sparsity
pattern contributes, even though the class note below describes the extra
entries as ignored.

```{doxygenclass} gko::preconditioner::Isai
:members:
```
