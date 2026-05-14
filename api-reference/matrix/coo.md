# `gko::matrix::Coo`

Coordinate format. Stores nonzeros as `(row, col, value)` triplets in
parallel arrays. Cheap to assemble and convert from, but typically
converted into CSR for performance-critical operations.

```{doxygenclass} gko::matrix::Coo
:members:
```
