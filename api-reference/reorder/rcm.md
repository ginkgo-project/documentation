# `gko::experimental::reorder::Rcm`

Reverse Cuthill–McKee bandwidth reduction. Builds a permutation that
clusters non-zeros near the diagonal — useful as a fill-in
pre-conditioner for sparse Cholesky / LU and as a cache-locality
hint for triangular solves.

```{doxygenclass} gko::experimental::reorder::Rcm
:members:
```
