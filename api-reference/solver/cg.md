# `gko::solver::Cg`

Conjugate gradient. The canonical Krylov method for symmetric
positive-definite systems. Each iteration costs one matrix-vector
product, one preconditioner apply, two inner products, and three AXPYs.

```{doxygenclass} gko::solver::Cg
:members:
```
