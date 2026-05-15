# `gko::experimental::factorization::Lu`

Sparse LU factorisation for general systems. Computes $L$ and $U$ such
that $A = L U$ and returns them as a `Factorization` in CSR format.
Requires the symbolic pattern to include every fill-in entry — for the
no-fill ILU(0) variant use [`Ilu`](ilu.md).

Lives in the `experimental` namespace pending API stabilisation.

```{doxygenclass} gko::experimental::factorization::Lu
:members:
```
