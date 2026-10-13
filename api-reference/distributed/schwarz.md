# `gko::experimental::distributed::preconditioner::`<br>`Schwarz`

Additive Schwarz preconditioner. Applies a user-supplied local solver
to each rank's local matrix; remote couplings are not iterated over,
which makes the operation embarrassingly parallel across ranks. The
factory also supports an L1 smoother variant (non-local absolute
row sums folded into the diagonal of the local matrix before generation) and an
algebraic coarse-grid correction for two-level Schwarz.

```{doxygenclass} gko::experimental::distributed::preconditioner::Schwarz
:members:
```
