# `gko::solver::Bicg`

Biconjugate gradient. Generalises CG to non-symmetric systems by
maintaining two coupled Krylov sequences (one with `A`, one with
`Aᵀ`). Cheap per iteration but the convergence is erratic — `Bicgstab`
is usually preferred in practice.

```{doxygenclass} gko::solver::Bicg
:members:
```
