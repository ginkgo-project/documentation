# `gko::stop`

Stopping criteria for iterative solvers. The iterative solvers in
[`gko::solver`](../solver/index.md) consume one or more `CriterionFactory`
objects through `with_criteria(...)`. Each factory produces a `Criterion`
whose `check()` decides whether to stop on each iteration. Multiple
criteria passed to `with_criteria` stop the solver as soon as any of them
fires.

## Base machinery

- [`Criterion`](criterion.md) — abstract base for all stopping
  criteria.
- [`Combined`](combined.md) — meta-criterion that stops when any of a
  list of inner criteria fires.

## Built-in criteria

- [`Iteration`](iteration.md) — fixed iteration cap.
- [`Time`](time.md) — wall-clock cap.
- [`ResidualNorm`](residual-norm.md) — explicit and implicit
  residual-norm criteria for solver tolerance tests, with three
  baseline modes (right-hand-side, initial-residual, absolute).

## Helper functions

The quickest way to configure stopping is to pass these functions to
`with_criteria`; the value type is taken from the solver's vectors:

- `max_iters(n)` — stop after `n` iterations ([`Iteration`](iteration.md)).
- `min_iters(n, criteria...)` — check the given criteria only after `n`
  iterations have finished ([`Iteration`](iteration.md)).
- `time_limit(duration)` — wall-clock cap ([`Time`](time.md)).
- `relative_residual_norm(tol)`, `initial_residual_norm(tol)`,
  `absolute_residual_norm(tol)` and their `*_implicit_residual_norm`
  variants ([`ResidualNorm`](residual-norm.md)).

```cpp
auto factory =
    gko::solver::Cg<double>::build()
        .with_criteria(
            gko::stop::max_iters(1000u),
            gko::stop::min_iters(10u, gko::stop::relative_residual_norm(1e-10)))
        .on(exec);
```

```{toctree}
:hidden:
:maxdepth: 1

criterion
combined
iteration
time
residual-norm
```
