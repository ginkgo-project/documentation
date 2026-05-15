# `gko::batch::solver::Bicgstab`

Batched stabilised biconjugate gradient. The batched analogue of
[`gko::solver::Bicgstab`](../solver/bicgstab.md): suitable for batches
of non-symmetric systems where each item benefits from BiCGSTAB's
short recurrence and avoidance of an explicit `A^H` apply. Each batch
item runs its own iteration with its own convergence test.

```{doxygenclass} gko::batch::solver::Bicgstab
:members:
```
