# `gko::solver::UpperTrs`

Upper-triangular solve. Direct substitution against an upper-triangular
matrix `U`: `apply(b, x)` produces `x` such that `U x = b`. The
backward-sweep half of LU- and IC-based preconditioners; pairs with
`LowerTrs` to form a full triangular solve chain.

```{doxygenclass} gko::solver::UpperTrs
:members:
```
