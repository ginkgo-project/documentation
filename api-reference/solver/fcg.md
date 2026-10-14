# `gko::solver::Fcg`

Flexible conjugate gradient. A CG variant that tolerates a changing
preconditioner across iterations, which is necessary when the preconditioner
itself is inexact (inner iterative solve, randomized smoothers, …).
Costs one extra inner product per iteration over plain CG. Suited to
symmetric positive-definite matrices.

The update coefficient is computed as the Polak-Ribière form
$\langle r_k - r_{k-1}, z_k\rangle / \langle r_{k-1}, z_{k-1}\rangle$, the
choice that keeps the search directions conjugate under a changing
preconditioner, even though the class documentation below names
Fletcher-Reeves; see the concept page [FCG](../../user-guide/concepts/solvers/fcg.md).

```{doxygenclass} gko::solver::Fcg
:members:
```
