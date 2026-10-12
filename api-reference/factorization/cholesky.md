# `gko::experimental::factorization::`<br>`Cholesky`

Sparse Cholesky factorisation for symmetric positive-definite systems.
Computes the lower-triangular factor $L$ such that $A = L L^H$ and
returns it as a `Factorization` in CSR format. It computes the symbolic
factorization (the fill-in pattern of $L + L^H$) itself, or takes a
precomputed one through the `symbolic_factorization` parameter, which must
then contain every fill-in entry. For a factor restricted to the pattern of
$A$, use [`Ic`](ic.md) (IC(0)).

:::{attention}
This class lives in the `experimental` namespace; its interface may change.
:::

```{doxygenclass} gko::experimental::factorization::Cholesky
:members:
```
