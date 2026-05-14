# `gko::matrix::RowGatherer`

Gathers rows by index. Stores an array of source-row indices;
`apply(b, x)` writes `x[i] = b[row_indices[i]]`. The communication
primitive behind the off-diagonal SpMV in distributed matrices and a
useful tool for any rearrange-by-index workload.

```{doxygenclass} gko::matrix::RowGatherer
:members:
```
