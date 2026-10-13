# `gko::stop::Criterion`

Abstract base for all stopping criteria. A `CriterionFactory` generates
a criterion from `CriterionArgs` (the system matrix, the right-hand side
`b`, the initial guess `x` and the initial residual). The resulting
`Criterion` receives the data of each iteration through `update()` (iteration
count, residual, residual norm, implicit residual estimate, solution) and is
queried with `check()` on every iteration of the consuming solver.

```{doxygenclass} gko::stop::Criterion
:members:
```
