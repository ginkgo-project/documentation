# `gko::matrix::Identity`

Identity LinOp. `apply(b, x)` copies `b` into `x` without allocating
explicit storage for the identity matrix itself. Used as a no-op
preconditioner placeholder and in the SpGEAM path (`A->apply(alpha, I,
beta, B)` computes `B = alpha * A + beta * B`).

```{doxygenclass} gko::matrix::Identity
:members:
```
