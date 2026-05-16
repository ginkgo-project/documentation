# `gko::multigrid::MultigridLevel`

Abstract interface for a single level of an algebraic-multigrid
hierarchy. Exposes three `LinOp`s — the restriction $R$, the coarse
operator $A_c$, and the prolongation $P$ — so that
[`gko::solver::Multigrid`](../solver/multigrid.md) can drive any
generator that implements this interface without knowing how the level
was assembled.

Concrete level generators (`Pgm`, `FixedCoarsening`, …) implement
this interface. New generators typically combine the
`EnableMultigridLevel<ValueType>` CRTP mixin (which supplies the
storage and accessor boilerplate) with their own factorisation logic.

```{doxygenclass} gko::multigrid::MultigridLevel
:members:
```
