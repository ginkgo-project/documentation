# `gko::stop::ResidualNorm`

Residual-norm tolerance criteria. The `baseline` factory parameter
picks how the threshold is interpreted (relative to the right-hand
side, relative to the initial residual, or absolute) — see the class
docstring for the three modes.

Two flavours:

- `ResidualNorm` — uses the 2-norm of the residual, taken from the norm
  the solver provides or computed from the solver's residual vector with
  one global reduction per check. Solvers that pass neither (MINRES,
  Multigrid) make it compute $b - A x$, an extra SpMV per check.
- `ImplicitResidualNorm` — uses the squared-norm estimate the solver
  maintains internally (typically an inner product that is already
  computed each iteration), avoiding that reduction at the cost of a
  quantity that can drift from the true residual norm on long runs.

Helper functions build these criteria without spelling out the value type:
`relative_residual_norm(tol)` (baseline `rhs_norm`),
`initial_residual_norm(tol)` (`initial_resnorm`),
`absolute_residual_norm(tol)` (`absolute`), and
`relative_implicit_residual_norm`, `initial_implicit_residual_norm` and
`absolute_implicit_residual_norm` for `ImplicitResidualNorm`.

```{doxygenclass} gko::stop::ResidualNorm
:members:
```

```{doxygenclass} gko::stop::ImplicitResidualNorm
:members:
```

```{doxygenclass} gko::stop::ResidualNormBase
:members:
```

```{doxygenfunction} gko::stop::relative_residual_norm
```

```{doxygenfunction} gko::stop::initial_residual_norm
```

```{doxygenfunction} gko::stop::absolute_residual_norm
```

```{doxygenfunction} gko::stop::relative_implicit_residual_norm
```

```{doxygenfunction} gko::stop::initial_implicit_residual_norm
```

```{doxygenfunction} gko::stop::absolute_implicit_residual_norm
```
