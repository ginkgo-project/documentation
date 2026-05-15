# `gko::preconditioner::Isai`

Incomplete Sparse Approximate Inverse preconditioner. Computes a sparse
approximation $M_A \approx A^{-1}$ (or its triangular- / Cholesky-factor
variants) directly, so `apply` is a single sparse matrix-vector product
without any triangular solve. Useful when triangular substitution
becomes a serial bottleneck on highly-parallel hardware.

```{doxygenclass} gko::preconditioner::Isai
:members:
```
