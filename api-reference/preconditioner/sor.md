# `gko::preconditioner::Sor`

Successive Over-Relaxation preconditioner. Splits the matrix as
$A = D + L + U$ and applies a forward sweep weighted by the relaxation
factor $\omega \in (0, 2)$. The symmetric variant (SSOR) follows the
forward sweep with a backward sweep, yielding a symmetric
preconditioner suitable for symmetric Krylov solvers.

```{doxygenclass} gko::preconditioner::Sor
:members:
```
