# `gko::matrix::Dense`

Row-major dense matrix. Doubles as Ginkgo's vector type (a Dense
matrix with one column) and as the storage for right-hand sides and
solution vectors in iterative solvers. Backed by a contiguous value
array in which consecutive rows are `stride` elements apart.

```{doxygenclass} gko::matrix::Dense
:members:
```
