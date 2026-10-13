# `gko::matrix::SparsityCsr`

Sparsity pattern only — CSR row pointers and column indices, plus one
shared value (a value array of length 1) that stands in for every entry.
Used wherever an algorithm needs the structure of a matrix but not its
numerical entries: reordering, the symbolic pattern of LU and Cholesky
factorizations, and graph algorithms.

```{doxygenclass} gko::matrix::SparsityCsr
:members:
```
