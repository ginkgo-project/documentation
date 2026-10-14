# `gko::stop`

Stopping criteria for iterative solvers. Every Krylov solver in
[`gko::solver`](../solver/index.md) consumes a `CriterionFactory`
that produces a `Criterion` whose `check()` decides whether to stop
on each iteration. When a solver receives several criteria, it stops as
soon as any one of them fires.

## Base machinery

- [`Criterion`](criterion.md): abstract base for all stopping
  criteria.
- [`Combined`](combined.md): meta-criterion that stops when any of a
  list of inner criteria fires.

## Built-in criteria

- [`Iteration`](iteration.md): fixed iteration cap.
- [`Time`](time.md): wall-clock cap.
- [`ResidualNorm`](residual-norm.md): explicit and implicit
  residual-norm criteria for solver tolerance tests, with three
  baseline modes (right-hand-side, initial-residual, absolute).

```{toctree}
:hidden:
:maxdepth: 1

criterion
combined
iteration
time
residual-norm
```
