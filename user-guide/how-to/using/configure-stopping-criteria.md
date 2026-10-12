# Configure stopping criteria

Every iterative solver in Ginkgo takes one or more **stopping criterion
factories** through `with_criteria(...)`. The solver stops when **any**
criterion fires (a logical OR). This page covers the common
combinations and the choices behind them.

## The two most-used criteria

```cpp
auto factory = gko::solver::Cg<double>::build()
                   .with_criteria(gko::stop::max_iters(1000u),
                                  gko::stop::relative_residual_norm(1e-8))
                   .on(exec);   // stops at ||r|| <= 1e-8 ||b||
```

- `stop::Iteration` — hard cap on iterations. Always include one; it
  guards against runaway loops if convergence never triggers.
- `stop::ResidualNorm` — stops when
  $\lVert r_k \rVert \le \tau \, \beta$, where $\tau$ is
  `reduction_factor` and the baseline $\beta$ defaults to
  $\lVert b \rVert$ (`mode::rhs_norm`).

These two together — a max-iter cap and a relative residual target —
cover most use cases.

### Criterion helpers

The helper functions need neither an executor nor a value type, and can be passed straight to
`with_criteria`:

| Helper | Stops when |
|---|---|
| `max_iters(n)` | after `n` iterations |
| `relative_residual_norm(tau)` | $\lVert r \rVert \le \tau \, \lVert b \rVert$ |
| `initial_residual_norm(tau)` | $\lVert r \rVert \le \tau \, \lVert r_0 \rVert$ |
| `absolute_residual_norm(tau)` | $\lVert r \rVert \le \tau$ |
| `relative_implicit_residual_norm(tau)`, `initial_implicit_residual_norm(tau)`, `absolute_implicit_residual_norm(tau)` | the same baselines, using the solver's implicit norm estimate |
| `time_limit(d)` | the wall-clock time `d` has elapsed |
| `min_iters(n, criteria...)` | `criteria...` are checked only after `n` iterations |

`min_iters` wraps the criteria that must wait for a minimum amount of work:

```cpp
auto solver_factory =
    gko::solver::Gmres<double>::build()
        .with_criteria(
            gko::stop::min_iters(10u, gko::stop::relative_residual_norm(1e-10)),
            gko::stop::max_iters(1000u))
        .on(exec);
```

The explicit builders (`Iteration::build().with_max_iters(...)`, `ResidualNorm<ValueType>::build()...`)
produce the same criteria, and the examples below use them where the builder
parameters matter.

## Relative vs absolute residual

```cpp
using BaseMode = gko::stop::mode;
gko::stop::ResidualNorm<double>::build()
    .with_baseline(BaseMode::rhs_norm)          // default: relative to ||b||
    .with_reduction_factor(1e-8)
    .on(exec);

gko::stop::ResidualNorm<double>::build()
    .with_baseline(BaseMode::initial_resnorm)   // relative to ||r_0||
    .with_reduction_factor(1e-8)
    .on(exec);

gko::stop::ResidualNorm<double>::build()
    .with_baseline(BaseMode::absolute)          // absolute: ||r_k|| <= tol
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

Pick `rhs_norm` when you want a tolerance that's independent of the
initial guess. The `initial_resnorm` baseline is fixed once per solve
to $\lVert b - A x_0 \rVert$, so when you chain warm-started solves
and feed each previous solution back as $x_0$, $\lVert r_0 \rVert$
shrinks and the implied stopping target $\tau \, \lVert r_0 \rVert$
shrinks with it — sometimes well below what the solver can actually
reach in finite precision. `rhs_norm` fixes the baseline to
$\lVert b \rVert$, which stays constant across the chain.

## Implicit vs explicit residual

`ResidualNorm` uses the residual norm the solver hands over, or computes the 2-norm of the
solver's residual vector, which costs one global reduction per check.
`ImplicitResidualNorm` reads the squared-norm estimate the solver already maintains (CG, BiCG,
BiCGSTAB, CGS, FCG, MINRES, PipeCG), which saves that reduction.

```cpp
gko::stop::ResidualNorm<double>::build()
    .with_reduction_factor(1e-8)
    .with_baseline(gko::stop::mode::initial_resnorm)
    .on(exec);

gko::stop::ImplicitResidualNorm<double>::build()
    .with_reduction_factor(1e-8)
    .on(exec);
```

The estimate is updated by recurrence and can drift from the true residual on long runs, so
pair `ImplicitResidualNorm` with an iteration cap. Prefer `ResidualNorm` when the tolerance must
refer to the unpreconditioned residual. For MINRES and Multigrid, `ResidualNorm` computes
$b - A x$ at each check, which costs an extra SpMV per check.

## Combine criteria

`with_criteria` takes any number of factories. Stop on **whichever
fires first**:

```cpp
auto factory = gko::solver::Cg<double>::build()
                   .with_criteria(gko::stop::max_iters(500u),
                                  gko::stop::relative_residual_norm(1e-10),
                                  gko::stop::time_limit(std::chrono::seconds{30}))
                   .on(exec);
```

`stop::Time` (`time_limit`) ends the solve after a wall-clock budget — useful for
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
  use right preconditioning ($A M^{-1} y = b$, recover $x = M^{-1} y$);
  CG and FCG apply $M^{-1}$ to the residual. Regardless of placement,
  `ResidualNorm` always refers to the *unpreconditioned* residual
  $r = b - A x$, while `ImplicitResidualNorm` uses the solver's own
  estimate (a preconditioned quantity for CG-type solvers).

:::{seealso}
- [Stopping criteria](../../concepts/stopping-criteria.md) — the conceptual reference.
- [Log convergence](log-convergence.md) — collect per-iteration data while solving.
:::
