# `gko::batch::matrix::Ell`

Batched equal-length-rows matrix. Like the non-batched
[`gko::matrix::Ell`](../matrix/ell.md), pads every row to the longest
row's nonzero count; in the batched form the column-index array is
shared across items so only the per-item values differ.

```{doxygenclass} gko::batch::matrix::Ell
:members:
```
