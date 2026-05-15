# `gko::factorization::ParIlut`

Parallel threshold-based incomplete LU. The LU counterpart of
[`ParIct`](par-ict.md): each sweep refines both the sparsity patterns
of $L$ and $U$ and their numerical entries, alternating add-fill-in,
fixed-point iteration, and drop-smallest steps. Algorithm of Anzt
et al.

```{doxygenclass} gko::factorization::ParIlut
:members:
```
