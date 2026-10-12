# `gko::experimental::factorization::Lu`

Sparse LU factorisation for general systems. Computes $L$ and $U$ such
that $A = L U$ and returns them as a `Factorization` in CSR format.
It computes the symbolic factorization itself (the algorithm is chosen by
`symbolic_algorithm`), or takes a precomputed one through
`symbolic_factorization`, which must contain every fill-in entry.
`symbolic_type::symmetric` is only valid for matrices with a symmetric
sparsity pattern. For the no-fill
ILU(0) variant use [`Ilu`](ilu.md).

:::{attention}
This class lives in the `experimental` namespace; its interface may change.
:::

```{doxygenclass} gko::experimental::factorization::Lu
:members:
```
