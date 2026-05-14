# `gko::matrix::SparsityCsr`

Sparsity pattern only — CSR row pointers and column indices, no values.
Used wherever an algorithm needs the structure of a matrix but not its
numerical entries: reordering, fill-in analysis, graph algorithms, and
the sparsity-reuse path for repeated SpGEMM.

```{doxygenclass} gko::matrix::SparsityCsr
:members:
```
