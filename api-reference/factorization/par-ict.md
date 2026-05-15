# `gko::factorization::ParIct`

Parallel threshold-based incomplete Cholesky. Refines both the
sparsity pattern of $L$ and its numerical entries simultaneously: at
each sweep it adds fill-in candidates from the current residual,
re-runs a fixed-point iteration, and drops the smallest entries by
magnitude. Algorithm of Anzt et al. (ParILUT family).

```{doxygenclass} gko::factorization::ParIct
:members:
```
