# `gko::solver::Chebyshev`

Chebyshev iteration. A non-Krylov stationary method that uses a
two-term recurrence with parameters derived from an estimate of the
spectrum of the (preconditioned) operator. No inner products — useful
as a smoother in multigrid and as an outer iteration when global
reductions are expensive.

```{doxygenclass} gko::solver::Chebyshev
:members:
```
