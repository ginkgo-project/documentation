# `gko::stop::Criterion`

Abstract base for all stopping criteria. The factory takes the system
matrix, the right-hand side, the initial guess, and any implicit
residual estimate; the resulting `Criterion` is queried with `check()`
on every iteration of the consuming solver.

```{doxygenclass} gko::stop::Criterion
:members:
```
