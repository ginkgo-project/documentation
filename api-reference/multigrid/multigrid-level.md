# `gko::multigrid::MultigridLevel`

Abstract interface for a single level of an algebraic-multigrid
hierarchy. Exposes the fine operator $A$ together with three `LinOp`s
(the restriction $R$, the coarse operator $A_c$, and the prolongation
$P$) so that
[`gko::solver::Multigrid`](../solver/multigrid.md) can drive any
generator that implements this interface without knowing how the level
was assembled.

Concrete level generators (`Pgm`, `FixedCoarsening`, …) implement
this interface. New generators derive from `LinOp` (for example through
`EnableLinOp`) together with the `EnableMultigridLevel<ValueType>`
mixin, which supplies the storage and accessor boilerplate; the
generator then calls `set_multigrid_level` with its operators.

```{doxygenclass} gko::multigrid::MultigridLevel
:members:
```
