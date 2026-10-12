# `gko::solver::Ir`

Iterative refinement. Approximately solves the residual equation
$A e = r$ with an inner solver, then updates the iterate with the
relaxation factor $\alpha$. With $\alpha = 1$ this is classical
iterative refinement; with $\alpha \ne 1$ and the identity inner solver
it is Richardson iteration.

Each inner solve starts from a zero initial guess by default
(`inner_initial_guess = initial_guess_mode::zero`); `initial_guess_mode::rhs`
starts it from the residual instead, which can save inner iterations when the
eigenvalues of $A$ are close to 1 but makes IR diverge when they are far from 1.

```{doxygenclass} gko::solver::Ir
:members:
```
