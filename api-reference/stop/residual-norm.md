# `gko::stop::ResidualNorm`

Residual-norm tolerance criteria. The three concrete variants:

- `AbsoluteResidualNorm` — stops when $\|r_k\|_2 \le \mathrm{tol}$.
- `RelativeResidualNorm` — stops when $\|r_k\|_2 \le \mathrm{tol}\, \|r_0\|_2$.
- `ImplicitResidualNorm` — uses the residual norm estimate the solver
  maintains internally (typically a square-root of an inner product
  that is already computed each iteration) instead of an explicit
  $\|b - A x_k\|$ computation. Cheapest of the three for solvers that
  expose such an estimate.

```{doxygenclass} gko::stop::ResidualNorm
:members:
```

```{doxygenclass} gko::stop::ResidualNormBase
:members:
```

```{doxygenclass} gko::stop::AbsoluteResidualNorm
:members:
```

```{doxygenclass} gko::stop::RelativeResidualNorm
:members:
```

```{doxygenclass} gko::stop::ImplicitResidualNorm
:members:
```
