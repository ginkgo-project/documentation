# `gko::experimental::factorization::Lu`

Sparse LU factorisation for general systems. Computes $L$ and $U$ such
that $A = L U$ and returns them as a `Factorization` in CSR format.
The symbolic factorization is computed first when the
`symbolic_factorization` parameter is not given, so that parameter is
optional: the default `symbolic_type::general` algorithm handles every
sparsity pattern, even though the class documentation below still lists a
symmetric pattern as a requirement. A pattern you supply must contain every
fill-in entry. For the no-fill ILU(0) variant use [`Ilu`](ilu.md).

:::{attention}
`symbolic_type::symmetric` is valid only for matrices with a symmetric
sparsity pattern; any other matrix is undefined behaviour and is likely to
crash the application. `symbolic_type::near_symmetric` is correct for every
matrix, performs best for almost symmetric patterns, and may use excessive
memory and time on others.
:::

Lives in the `experimental` namespace pending API stabilisation.

```{doxygenclass} gko::experimental::factorization::Lu
:members:
```
