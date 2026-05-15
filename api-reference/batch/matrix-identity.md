# `gko::batch::matrix::Identity`

Batched identity LinOp. The batched analogue of
[`gko::matrix::Identity`](../matrix/identity.md): `apply(b, x)` copies
the right-hand-side block into the solution block for every batch
item. Used as a no-op preconditioner placeholder.

```{doxygenclass} gko::batch::matrix::Identity
:members:
```
