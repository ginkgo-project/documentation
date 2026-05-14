# `gko::solver::PipeCg`

Pipelined conjugate gradient. Restructures the standard CG loop so that
the global reductions (inner products) can overlap with the next
matrix-vector product and preconditioner apply. Reduces synchronisation
latency on large distributed runs at the cost of more local work per
iteration.

```{doxygenclass} gko::solver::PipeCg
:members:
```
