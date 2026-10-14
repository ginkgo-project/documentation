# `gko::factorization::ParIc`

Parallel IC(0). Computes the incomplete Cholesky factor by iterating
an asynchronous fixed-point update on $L$ (the algorithm of Chow &
Patel) and converges to a factorisation that matches $A$ on its
sparsity pattern. Designed for fine-grained parallelism on GPUs and
many-core CPUs.

```{doxygenclass} gko::factorization::ParIc
:members:
```
