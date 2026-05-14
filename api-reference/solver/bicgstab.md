# `gko::solver::Bicgstab`

Stabilised biconjugate gradient. Smooths Bicg's erratic convergence by
combining each step with a GMRES(1)-style local minimisation. Avoids
the need to apply `Aᵀ` — only one matvec with `A` plus the
preconditioner per iteration — making it the practical default for
non-symmetric problems.

```{doxygenclass} gko::solver::Bicgstab
:members:
```
