# `gko::experimental::distributed::preconditioner::`<br>`Schwarz`

Additive Schwarz preconditioner. Applies a user-supplied local solver
to each rank's diagonal block; remote couplings are not iterated over,
which makes the operation embarrassingly parallel across ranks. The
factory also supports an L1 smoother variant (off-diagonal absolute
row sums folded into the local diagonal before generation) and an
algebraic coarse-grid correction for two-level Schwarz.

```{doxygenclass} gko::experimental::distributed::preconditioner::Schwarz
:members:
```
