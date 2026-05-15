# `gko::batch::solver::Cg`

Batched conjugate gradient. Runs an independent CG iteration for each
batch item on the same executor, sharing kernel launches across items.
Each item terminates on its own once its `residual_norm` falls below
the tolerance; the batch-wide solve completes when every item has
either converged or hit the `max_iterations` cap.

```{doxygenclass} gko::batch::solver::Cg
:members:
```
