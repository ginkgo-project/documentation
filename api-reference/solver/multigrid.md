# `gko::solver::Multigrid`

Multigrid framework. A composable solver that accepts a coarsening
strategy (e.g. `gko::multigrid::Pgm`), per-level smoothers, and a
coarse-grid solver, and runs V/W/F cycles between them. Used both as a
standalone solver and as a preconditioner inside Krylov methods.

```{doxygenclass} gko::solver::Multigrid
:members:
```
