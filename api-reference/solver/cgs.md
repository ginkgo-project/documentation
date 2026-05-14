# `gko::solver::Cgs`

Conjugate gradient squared. Replaces Bicg's `Aᵀ` apply with another
apply of `A`, doubling the per-iteration residual reduction but
amplifying numerical noise. Cheap per iteration but less robust than
`Bicgstab`; useful when `A` is well conditioned.

```{doxygenclass} gko::solver::Cgs
:members:
```
