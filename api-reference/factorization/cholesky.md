# `gko::experimental::factorization::`<br>`Cholesky`

Sparse Cholesky factorisation for symmetric positive-definite systems.
Computes the lower-triangular factor $L$ such that $A = L L^H$ and
returns it as a `Factorization` in CSR format. The symbolic
factorization is computed first when the `symbolic_factorization` parameter
is not given; a pattern you supply must contain every fill-in entry. For
the pattern of $A$ itself, without fill-in, use [`Ic`](ic.md) (IC(0)).

Lives in the `experimental` namespace pending API stabilisation.

```{doxygenclass} gko::experimental::factorization::Cholesky
:members:
```
