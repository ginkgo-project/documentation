# `gko::solver::Ir`

Iterative refinement. Approximately solves the residual equation
$A e = r$ with an inner solver, then updates the iterate with the
relaxation factor $\alpha$. With $\alpha = 1$ this is classical
iterative refinement; with $\alpha \ne 1$ and the identity inner solver
it is Richardson iteration.

```{doxygenclass} gko::solver::Ir
:members:
```
