# `gko::matrix::RowGatherer`

Gathers rows by index within a single process. Stores an array of
source-row indices; `apply(b, x)` writes `x[i] = b[row_idxs[i]]`, so it
serves any rearrange-by-index workload. The distributed matrix holds a
separate, communicating [`RowGatherer`](../distributed/row-gatherer.md)
to collect the off-process rows it needs for the non-local SpMV.

```{doxygenclass} gko::matrix::RowGatherer
:members:
```
