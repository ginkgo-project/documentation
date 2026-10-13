# `gko::solver::Bicgstab`

Stabilised biconjugate gradient. Smooths Bicg's erratic convergence by
combining each step with a GMRES(1)-style local minimisation. Avoids
the need to apply `Aᵀ` — two matvecs with `A` plus two
preconditioner applies per iteration — making it a common choice for
non-symmetric problems when the memory of GMRES is a concern.

```{doxygenclass} gko::solver::Bicgstab
:members:
```
