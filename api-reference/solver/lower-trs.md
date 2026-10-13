# `gko::solver::LowerTrs`

Lower-triangular solve. Direct (non-iterative) substitution against a
lower-triangular matrix `L`: `apply(b, x)` produces `x` such that
`L x = b`. The forward-sweep half of LU- and IC-based preconditioners,
and the kernel behind incomplete-factorisation solver chains. The
`algorithm` factory parameter (`sparselib` or `syncfree`) selects the
implementation on the CUDA executor.

```{doxygenclass} gko::solver::LowerTrs
:members:
```
