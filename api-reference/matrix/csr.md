# `gko::matrix::Csr`

Compressed sparse row format. Stores each row's nonzeros as a contiguous
slice of value and column-index arrays, with a row-pointer array
locating the start of each row. The general-purpose default for sparse
matrices in Ginkgo.

```{doxygenclass} gko::matrix::Csr
:members:
```
