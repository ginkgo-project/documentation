# `gko::solver::PipeCg`

Pipelined conjugate gradient. Restructures the standard CG loop so that
the global reductions (inner products) can overlap with the next
matrix-vector product and preconditioner apply. Reduces synchronisation
latency on large distributed runs at the cost of more local work and
extra vector storage per iteration. The rearranged operations make it less
numerically stable than standard CG, and like CG it targets symmetric
positive-definite matrices.

```{doxygenclass} gko::solver::PipeCg
:members:
```
