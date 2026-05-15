# `gko::factorization::ParIlu`

Parallel ILU(0). Computes the incomplete LU factors $L$ and $U$ by
iterating an asynchronous fixed-point update — the algorithm of Chow &
Patel — and converges to a factorisation that matches $A$ on its
sparsity pattern. Designed for fine-grained parallelism on GPUs and
many-core CPUs.

```{doxygenclass} gko::factorization::ParIlu
:members:
```
