# `gko::matrix::Ell`

Equal-length rows. Pads every row to the longest row's nonzero count
and stores values/columns in a regular 2-D layout. Memory-efficient
only for matrices with a uniform nonzero distribution; well suited to
GPU SpMV with predictable access patterns.

```{doxygenclass} gko::matrix::Ell
:members:
```
