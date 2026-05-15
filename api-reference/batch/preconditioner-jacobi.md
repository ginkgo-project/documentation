# `gko::batch::preconditioner::Jacobi`

Batched block-Jacobi preconditioner. Inverts the per-item diagonal
blocks of every batch entry and stores the inverses for re-use on each
`apply`. Block size is uniform across items. When the block size is
1 the preconditioner degenerates to pointwise diagonal scaling — the
common batched-iterative-solve default.

```{doxygenclass} gko::batch::preconditioner::Jacobi
:members:
```
