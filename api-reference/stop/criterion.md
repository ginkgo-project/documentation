# `gko::stop::Criterion`

Abstract base for all stopping criteria. The factory (a
`CriterionFactory`) takes the system matrix, the right-hand side, the
initial guess and, optionally, the initial residual (`CriterionArgs`). The
resulting `Criterion` is queried every iteration through `update()`, which
collects the iteration count, residual, residual norm, implicit squared
residual norm and solution, followed by `check()`. Custom criteria derive
from `Criterion` and override `check_impl`.

```{doxygenclass} gko::stop::Criterion
:members:
```
