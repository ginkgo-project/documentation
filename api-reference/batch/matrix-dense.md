# `gko::batch::matrix::Dense`

Batched dense matrix. Each item is a row-major dense block; storage
is a single contiguous values array offset by
`batch_id * num_rows * num_cols`. Use for batches of small dense
systems whose dimensions are too small to fan out cuBLAS / cuSPARSE
calls profitably.

```{doxygenclass} gko::batch::matrix::Dense
:members:
```
