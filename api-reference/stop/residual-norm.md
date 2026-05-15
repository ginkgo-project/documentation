# `gko::stop::ResidualNorm`

Residual-norm tolerance criteria. The `baseline` factory parameter
picks how the threshold is interpreted (relative to the right-hand
side, relative to the initial residual, or absolute) — see the class
docstring for the three modes.

Two flavours:

- `ResidualNorm` — uses the explicit residual $\|r_k\|_2 = \|b - A x_k\|_2$.
- `ImplicitResidualNorm` — uses the residual-norm estimate the solver
  maintains internally (typically the square root of an inner product
  that is already computed each iteration), avoiding the extra
  matrix-vector product and global reduction at the cost of a quantity
  that can drift from the true residual norm on long runs.

```{doxygenclass} gko::stop::ResidualNorm
:members:
```

```{doxygenclass} gko::stop::ImplicitResidualNorm
:members:
```

```{doxygenclass} gko::stop::ResidualNormBase
:members:
```
