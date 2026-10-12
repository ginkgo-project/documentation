# `gko::matrix::RowGatherer`

Gathers rows by index. Stores an array of source-row indices;
`apply(b, x)` writes `x[i] = b[row_indices[i]]`, a useful tool for any
rearrange-by-index workload on a single process. The halo exchange of
distributed matrices uses the separate
[`experimental::distributed::RowGatherer`](../distributed/row-gatherer.md).

```{doxygenclass} gko::matrix::RowGatherer
:members:
```
