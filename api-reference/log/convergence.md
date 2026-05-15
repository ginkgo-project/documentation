# `gko::log::Convergence`

The simplest solver-facing logger. Records the iteration count at
which a solver converged (or stopped) and the residual norm at that
point. Attach to any iterative solver to read back these two numbers
after `apply`.

```{doxygenclass} gko::log::Convergence
:members:
```
