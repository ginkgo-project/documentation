# `gko::stop`

Stopping criteria for iterative solvers. Every Krylov solver in
[`gko::solver`](../solver/index.md) consumes a `Criterion::Factory`
that produces a `Criterion` whose `check()` decides whether to stop
on each iteration. Multiple criteria can be combined.

## Base machinery

- [`Criterion`](criterion.md) — abstract base for all stopping
  criteria.
- [`Combined`](combined.md) — meta-criterion that stops when any of a
  list of inner criteria fires.

## Built-in criteria

- [`Iteration`](iteration.md) — fixed iteration cap.
- [`Time`](time.md) — wall-clock cap.
- [`ResidualNorm`](residual-norm.md) — the explicit residual-norm
  family (`AbsoluteResidualNorm`, `RelativeResidualNorm`,
  `ImplicitResidualNorm`) for solver tolerance tests.
- [`ResidualNormReduction`](residual-norm-reduction.md) — deprecated
  alias that stops on `‖r_k‖ / ‖r_0‖ < tol`; prefer the explicit
  variants in `ResidualNorm` instead.

```{toctree}
:hidden:
:maxdepth: 1

criterion
combined
iteration
time
residual-norm
residual-norm-reduction
```
