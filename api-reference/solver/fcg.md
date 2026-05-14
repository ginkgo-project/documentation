# `gko::solver::Fcg`

Flexible conjugate gradient. A CG variant that tolerates a changing
preconditioner across iterations — necessary when the preconditioner
itself is inexact (inner iterative solve, randomized smoothers, …).
Costs one extra inner product per iteration over plain CG.

```{doxygenclass} gko::solver::Fcg
:members:
```
