# Configure stopping criteria

Every iterative solver in Ginkgo takes one or more **stopping criterion
factories** through `with_criteria(...)`. The solver stops when **any**
criterion fires (a logical OR). This page covers the common
combinations and the choices behind them.

## The two most-used criteria

```cpp
.with_criteria(
    gko::stop::Iteration::build()
        .with_max_iters(1000u).on(exec),
    gko::stop::ResidualNorm<ValueType>::build()
        .with_reduction_factor(1e-8).on(exec))
```

- `stop::Iteration` — hard cap on iterations. Always include one; it
  guards against runaway loops if convergence never triggers.
- `stop::ResidualNorm` — relative residual reduction. Stops when
  $\lVert r_k \rVert \le \tau \, \lVert r_0 \rVert$ where $\tau$ is
  `reduction_factor`.

These two together — a max-iter cap and a relative residual target —
cover most use cases.

## Relative vs absolute residual

```cpp
using BaseMode = gko::stop::mode;
gko::stop::ResidualNorm<double>::build()
    .with_baseline(BaseMode::initial_resnorm)   // default: relative to ||r_0||
    .with_reduction_factor(1e-8)
    .on(exec);

gko::stop::ResidualNorm<double>::build()
    .with_baseline(BaseMode::absolute)          // absolute: ||r_k|| <= tol
    .with_reduction_factor(1e-8)
    .on(exec);

gko::stop::ResidualNorm<double>::build()
    .with_baseline(BaseMode::rhs_norm)          // relative to ||b||
    .with_reduction_factor(1e-8)
    .on(exec);
```

`reduction_factor` is the value compared against the **chosen
baseline**:

| `baseline` | Stops when |
|---|---|
| `initial_resnorm` *(default)* | $\lVert r_k \rVert \le \tau \, \lVert r_0 \rVert$ |
| `rhs_norm` | $\lVert r_k \rVert \le \tau \, \lVert b \rVert$ |
| `absolute` | $\lVert r_k \rVert \le \tau$ |

Pick `rhs_norm` when you want a tolerance that's independent of how
good your initial guess is — useful when chaining solves with warm
starts.

## Implicit vs explicit residual

Some Krylov solvers compute the residual norm **implicitly** as a
by-product of their inner-product reductions; that costs nothing
extra. Explicit residual computation runs an additional SpMV per
iteration.

```cpp
gko::stop::ResidualNorm<double>::build()
    .with_reduction_factor(1e-8)
    .with_baseline(gko::stop::mode::initial_resnorm)
    .on(exec);

gko::stop::ImplicitResidualNorm<double>::build()
    .with_reduction_factor(1e-8)
    .on(exec);
```

`ImplicitResidualNorm` is cheap — it reads the value the solver
already computed. `ResidualNorm` is explicit — it recomputes
$r = b - A x$. On matrix-bandwidth-bound iterative solvers, the
explicit version roughly doubles per-iteration cost. Prefer
`ImplicitResidualNorm` unless you specifically need the
unpreconditioned residual.

## Combine criteria

`with_criteria` takes any number of factories. Stop on **whichever
fires first**:

```cpp
.with_criteria(
    gko::stop::Iteration::build().with_max_iters(500u).on(exec),
    gko::stop::ResidualNorm<double>::build()
        .with_reduction_factor(1e-10).on(exec),
    gko::stop::Time::build()
        .with_time_limit(std::chrono::seconds{30}).on(exec))
```

`stop::Time` ends the solve after a wall-clock budget — useful for
real-time / latency-bounded applications.

## Custom criteria

When the built-ins don't fit, subclass `gko::stop::Criterion` and
provide a `Factory`. The `check` method receives the current iterate
and decides whether to stop. See `include/ginkgo/core/stop/criterion.hpp`
for the interface.

## Common pitfalls

- **One `with_criteria` call overrides previous ones.** Pass all
  criteria in a single `with_criteria(...)` invocation, not multiple.
- **Don't forget the iteration cap.** Without one, a non-convergent
  problem loops forever.
- **Preconditioner choice changes the residual.** GMRES and BiCGSTAB
  preconditioner the right side ($A M^{-1} y = b$); the criterion
  always sees the *unpreconditioned* residual $r = b - A x$ regardless
  of placement.

:::{seealso}
- [Stopping criteria](../../concepts/stopping-criteria.md) — the conceptual reference.
- [Log convergence](log-convergence.md) — collect per-iteration data while solving.
:::
