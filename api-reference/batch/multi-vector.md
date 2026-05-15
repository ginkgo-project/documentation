# `gko::batch::MultiVector`

The right-hand-side and solution container for batched solvers. Holds
one dense block per batch item, with every item sharing the same
`(num_rows, num_rhs)` shape. Backed by a single contiguous values
array indexed by `batch_id * num_rows * num_rhs`.

```{doxygenclass} gko::batch::MultiVector
:members:
```
