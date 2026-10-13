# `gko::experimental::reorder::Rcm`

Reverse Cuthill–McKee bandwidth reduction. Builds a permutation that
clusters non-zeros near the diagonal — useful as a fill-in
pre-conditioner for sparse Cholesky / LU and as a cache-locality
hint for triangular solves. The class template has one parameter,
the index type (`Rcm<int>`); the older `gko::reorder::Rcm<ValueType, IndexType>`
is deprecated in favour of it.

```{doxygenclass} gko::experimental::reorder::Rcm
:members:
```
