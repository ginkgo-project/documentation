# `gko::batch::solver::BatchSolver`

Base class for every batched solver. Holds the system matrix and
preconditioner pointers and exposes the accessors shared across all
batched solvers: the runtime setters `reset_tolerance`,
`reset_max_iterations` and `reset_tolerance_type`, the getters
`get_tolerance`, `get_max_iterations` and `get_tolerance_type`, and
`get_system_matrix` and `get_preconditioner`. A different preconditioner
is supplied when the solver is generated, through the factory parameters
`preconditioner` or `generated_preconditioner`. Derived solvers plug in
the iteration kernel.

```{doxygenclass} gko::batch::solver::BatchSolver
:members:
```
