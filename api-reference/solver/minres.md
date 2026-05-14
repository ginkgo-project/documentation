# `gko::solver::Minres`

Minimum residual method for symmetric (possibly indefinite) systems.
Minimises the 2-norm of the residual over the current Krylov subspace
using a short three-term recurrence. The right choice when the matrix
is symmetric but not guaranteed positive definite.

```{doxygenclass} gko::solver::Minres
:members:
```
