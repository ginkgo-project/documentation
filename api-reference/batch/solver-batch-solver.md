# `gko::batch::solver::BatchSolver`

Base class for every batched solver. Holds the system matrix and
preconditioner pointers and exposes the runtime controls shared across
all batched solvers: `reset_tolerance`, `reset_max_iterations`,
`reset_tolerance_type` and the matching getters, plus `get_system_matrix`
and `get_preconditioner`. Derived solvers plug in the iteration kernel.

```{doxygenclass} gko::batch::solver::BatchSolver
:members:
```
