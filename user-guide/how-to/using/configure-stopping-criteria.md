# Configure stopping criteria

Every iterative solver in Ginkgo takes one or more **stopping criterion
factories** through `with_criteria(...)`. When you pass more than one,
they are combined through `gko::stop::Combined`, which always stops the
solver as soon as **any** criterion is met (a logical OR). This page covers the common
combinations and the choices behind them.

## The two most-used criteria

```cpp
auto solver_factory = gko::solver::Cg<ValueType>::build()
    .with_criteria(
        gko::stop::Iteration::build()
            .with_max_iters(1000u).on(exec),
        gko::stop::ResidualNorm<ValueType>::build()
            .with_reduction_factor(1e-8).on(exec))
    .on(exec);
```

- `stop::Iteration`: hard cap on iterations. Always include one; it
  guards against runaway loops if convergence never triggers.
- `stop::ResidualNorm`: relative residual reduction. Stops when
  $\lVert r_k \rVert \le \tau \, \lVert b \rVert$ where $\tau$ is
  `reduction_factor` (the default `baseline` is `rhs_norm`).

These two together (a max-iter cap and a relative residual target)
cover most use cases.

## Relative vs absolute residual

```cpp
using BaseMode = gko::stop::mode;
gko::stop::ResidualNorm<double>::build()
    .with_baseline(BaseMode::initial_resnorm)   // relative to ||r_0||
    .with_reduction_factor(1e-8)
    .on(exec);

gko::stop::ResidualNorm<double>::build()
    .with_baseline(BaseMode::absolute)          // absolute: ||r_k|| <= tol
    .with_reduction_factor(1e-8)
    .on(exec);

gko::stop::ResidualNorm<double>::build()
    .with_baseline(BaseMode::rhs_norm)          // default: relative to ||b||
    .with_reduction_factor(1e-8)
    .on(exec);
```

`reduction_factor` is the value compared against the **chosen
baseline**:

| `baseline` | Stops when |
|---|---|
| `rhs_norm` *(default)* | $\lVert r_k \rVert \le \tau \, \lVert b \rVert$ |
| `initial_resnorm` | $\lVert r_k \rVert \le \tau \, \lVert r_0 \rVert$ |
| `absolute` | $\lVert r_k \rVert \le \tau$ |

The default `rhs_norm` gives a tolerance that is independent of the
initial guess. The `initial_resnorm` baseline is fixed once per solve
to $\lVert b - A x_0 \rVert$, so when you chain warm-started solves
and feed each previous solution back as $x_0$, $\lVert r_0 \rVert$
shrinks and the implied stopping target $\tau \, \lVert r_0 \rVert$
shrinks with it, sometimes well below what the solver can actually
reach in finite precision. `rhs_norm` keeps the baseline at
$\lVert b \rVert$, which stays constant across the chain.

## Implicit vs explicit residual

Most Krylov solvers maintain a residual through their recurrences and
hand it to the criterion, so a residual-norm check costs one norm
computation and no extra SpMV (MINRES and Multigrid are the exceptions
described below). Some solvers also keep a squared
residual estimate as a by-product of their inner products, which
`ImplicitResidualNorm` reads.

```cpp
gko::stop::ResidualNorm<double>::build()
    .with_reduction_factor(1e-8)
    .with_baseline(gko::stop::mode::rhs_norm)
    .on(exec);

gko::stop::ImplicitResidualNorm<double>::build()
    .with_reduction_factor(1e-8)
    .on(exec);
```

`ResidualNorm` uses the norm the solver passes to the criterion, or the
2-norm of the residual vector the solver passes (one reduction). Solvers
such as CG, BiCGSTAB and CGS pass their recurrence residual; GMRES passes
its Givens-rotation estimate. The criterion computes $r = b - A x$ with
a matrix application only as a fallback, for solvers that pass neither
(for example MINRES and Multigrid).

`ImplicitResidualNorm` reads the squared residual estimate from the
solver's recurrence. For CG-type solvers it saves the global reduction
that the norm of the residual vector needs. With a preconditioner the
estimate is $M^{-1}$-weighted, a different measure from
$\lVert b - A x \rVert$. Prefer `ResidualNorm` when the tolerance is
stated for the residual of the system, and `ImplicitResidualNorm` for
loose inner solves where the reduction saving matters.

## Combine criteria

`with_criteria` takes any number of factories. The solver stops as soon
as **any** of them is met:

```cpp
auto solver_factory = gko::solver::Cg<double>::build()
    .with_criteria(
        gko::stop::Iteration::build().with_max_iters(500u).on(exec),
        gko::stop::ResidualNorm<double>::build()
            .with_reduction_factor(1e-10).on(exec),
        gko::stop::Time::build()
            .with_time_limit(std::chrono::seconds{30}).on(exec))
    .on(exec);
```

`stop::Time` ends the solve after a wall-clock budget, useful for
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
- **Preconditioner choice changes the implicit norm.** `ResidualNorm`
  measures the residual of the system, taken from the solver's
  recurrence (or the GMRES estimate). `ImplicitResidualNorm` reads the
  recurrence's squared estimate, which with a preconditioner is
  $M^{-1}$-weighted for CG-type solvers.

:::{seealso}
- [Stopping criteria](../../concepts/stopping-criteria.md): the conceptual reference.
- [Log convergence](log-convergence.md): collect per-iteration data while solving.
:::
