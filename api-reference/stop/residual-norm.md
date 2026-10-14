# `gko::stop::ResidualNorm`

Residual-norm tolerance criteria. The `baseline` factory parameter
picks how the threshold is interpreted (relative to the right-hand
side, relative to the initial residual, or absolute); see the class
docstring for the three modes.

Two flavours:

- `ResidualNorm`: tests the residual $\|r_k\|_2 = \|b - A x_k\|_2$.
  The norm comes from the residual vector or residual norm that the solver
  passes to the criterion; when the solver passes neither (MINRES,
  Multigrid), the criterion evaluates $b - A x_k$ with a matrix-vector
  product.
- `ImplicitResidualNorm`: uses the squared residual-norm estimate the
  solver maintains internally (typically an inner product that is already
  computed each iteration). For CG-type solvers this saves the global
  reduction that the norm of the residual vector needs, and it never needs a
  matrix-vector product, at the cost of a quantity that can drift from the
  true residual norm on long runs.

```{doxygenclass} gko::stop::ResidualNorm
:members:
```

```{doxygenclass} gko::stop::ImplicitResidualNorm
:members:
```

```{doxygenclass} gko::stop::ResidualNormBase
:members:
```
