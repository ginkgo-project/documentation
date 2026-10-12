# `gko::multigrid::MultigridLevel`

Abstract interface for a single level of an algebraic-multigrid
hierarchy. Exposes the fine operator $A$ and three derived `LinOp`s — the
restriction $R$, the coarse operator $A_c$, and the prolongation $P$ — so that
[`gko::solver::Multigrid`](../solver/multigrid.md) can drive any
generator that implements this interface without knowing how the level
was assembled.

Concrete level generators (`Pgm`, `Pmis`, `Rs`, `UniformCoarsening`,
`FixedCoarsening`) implement this interface. New generators derive from `LinOp`
and the `EnableMultigridLevel<ValueType>` mixin, which stores the four
operators (set through `set_multigrid_level`) and provides the composition
$P A_c R$ used as the level's `apply`.

```{doxygenclass} gko::multigrid::MultigridLevel
:members:
```
