# `gko::experimental::factorization::Cholesky`

Sparse Cholesky factorisation for symmetric positive-definite systems.
Computes the lower-triangular factor $L$ such that $A = L L^H$ and
returns it as a `Factorization` in CSR format. Requires the symbolic
sparsity pattern to include every fill-in entry — if you only have the
pattern of $A$ itself, use [`Ic`](ic.md) (IC(0)) instead.

Lives in the `experimental` namespace pending API stabilisation.

```{doxygenclass} gko::experimental::factorization::Cholesky
:members:
```
