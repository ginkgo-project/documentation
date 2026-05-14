# `gko::matrix::Diagonal`

Diagonal matrix. Stores one value per diagonal entry; `apply(b, x)`
performs element-wise multiplication. The natural representation for
diagonal scalings, row/column rescaling preconditioners, and the
diagonal preconditioner inside `gko::preconditioner::Jacobi` at block
size 1.

```{doxygenclass} gko::matrix::Diagonal
:members:
```
