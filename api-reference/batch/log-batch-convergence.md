# `gko::batch::log::BatchConvergence`

Batched analogue of [`gko::log::Convergence`](../log/convergence.md):
records the iteration count at which each batch item converged (or
stopped) and the residual norm at that point. Per-item arrays so you
can inspect, for instance, that 30 of 32 items converged in
fewer than 100 iterations and the remaining 2 hit the cap.

```{doxygenclass} gko::batch::log::BatchConvergence
:members:
```
