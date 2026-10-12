# `gko::matrix::SparsityCsr`

Sparsity pattern of a CSR matrix: row pointers and column indices plus a
single value shared by all nonzeros (1 by default), so `apply` acts as
the pattern's adjacency matrix. Used wherever an algorithm needs the
structure of a matrix: reordering, coarsening, and the
`symbolic_factorization` input of the experimental `Cholesky` and `Lu`
factorizations.

```{doxygenclass} gko::matrix::SparsityCsr
:members:
```
