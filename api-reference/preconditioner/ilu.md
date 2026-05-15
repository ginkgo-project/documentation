# `gko::preconditioner::Ilu`

Incomplete LU preconditioner. Given incomplete factors $L$ and $U$
(computed externally or via `gko::factorization::Ilu` /
`gko::factorization::ParIlu`), `apply(b, x)` solves $LU x = b$ by
chaining a lower-triangular solve with an upper-triangular solve. The
solvers for $L$ and $U$ are configurable, so either direct
substitution or an inner iterative solve can be used.

```{doxygenclass} gko::preconditioner::Ilu
:members:
```
