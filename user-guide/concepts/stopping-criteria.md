# Stopping criteria

An iterative solver needs a way to decide when to stop. Ginkgo's stopping criteria are composable factories that solvers consult each iteration. This page covers the built-in criteria, how they compose, and how to plug them into solver factories.

## The abstraction

A stopping criterion factory produces a runtime criterion object that the solver calls after each iteration. The criterion sees the current iteration count, the current residual, and the current solution vector; it returns a per-right-hand-side converged flag. When all right-hand sides are flagged, the solver stops.

Because criteria are factories — the same pattern as solvers and preconditioners — they plug into solver factories via `with_criteria(...)` and are constructed together with the solver at `generate` time. This keeps criterion configuration independent of the solver algorithm.

## Built-in criteria

Ginkgo ships four primary criteria:

```cpp
// Stop after a fixed number of iterations.
gko::stop::Iteration::build()
    .with_max_iters(1000u)
    .on(exec);

// Stop when the explicit residual norm drops below a fraction of the baseline.
gko::stop::ResidualNorm<double>::build()
    .with_baseline(gko::stop::mode::initial_resnorm)
    .with_reduction_factor(1e-8)
    .on(exec);

// Stop using the solver's internal recurrence-based residual estimate.
gko::stop::ImplicitResidualNorm<double>::build()
    .with_reduction_factor(1e-10)
    .on(exec);

// Stop after a wall-clock duration.
gko::stop::Time::build()
    .with_time_limit(std::chrono::seconds{30})
    .on(exec);
```

- **`Iteration`** — unconditionally stops after `max_iters` iterations. A safety net that prevents infinite loops regardless of convergence.
- **`ResidualNorm`** — stops when `‖r‖ / baseline ≤ reduction_factor`. The `baseline` can be `initial_resnorm` (relative to the initial residual), `rhs_norm` (relative to `‖b‖`), or `absolute` (no normalisation). This criterion requires an explicit SpMV to compute `r = b − A x` each time it is checked.
- **`ImplicitResidualNorm`** — uses the residual estimate maintained inside the Krylov recurrence. No extra SpMV, but the estimate can drift from the true residual over many iterations. See the *Implicit vs explicit residual norms* section below for the trade-off.
- **`Time`** — stops after a wall-clock duration regardless of convergence. Useful for anytime algorithms or hard latency budgets.

## Composition with `Combined`

Multiple criteria compose via `gko::stop::Combined`, which stops when *any* sub-criterion fires (OR semantics by default). Pass the sub-criteria as factories:

```cpp
auto criteria = gko::stop::Combined::build()
                    .with_criteria(
                        gko::stop::Iteration::build()
                            .with_max_iters(1000u)
                            .on(exec),
                        gko::stop::ResidualNorm<double>::build()
                            .with_reduction_factor(1e-8)
                            .on(exec))
                    .on(exec);
```

This stops when either 1000 iterations have run *or* the residual has reduced by a factor of 1e-8 — whichever comes first. Using at least an `Iteration` bound together with a `ResidualNorm` criterion is good practice: the iteration bound guards against divergence, while the norm criterion avoids unnecessary extra work after convergence.

## Plugging into a solver

`with_criteria` accepts a variadic pack of criterion factories; `Combined` is constructed implicitly when you pass more than one:

```cpp
auto solver_factory = gko::solver::Cg<double>::build()
                          .with_criteria(
                              gko::stop::Iteration::build()
                                  .with_max_iters(1000u)
                                  .on(exec),
                              gko::stop::ResidualNorm<double>::build()
                                  .with_reduction_factor(1e-8)
                                  .on(exec))
                          .on(exec);

auto solver = solver_factory->generate(system_matrix);
solver->apply(b, x);
```

If you already have a `Combined` factory, you can pass it directly:

```cpp
auto solver_factory = gko::solver::Cg<double>::build()
                          .with_criteria(criteria)
                          .on(exec);
```

## Inspecting why a solver stopped

Solvers expose convergence information through a logger. Attach `gko::log::Convergence` before calling `apply` to read the final iteration count and residual norm afterward:

```cpp
auto convergence = std::make_shared<gko::log::Convergence<double>>();
solver->add_logger(convergence);
solver->apply(b, x);

auto iters = convergence->get_num_iterations();
auto res   = convergence->get_residual_norm();
```

For batched solvers, use `gko::log::BatchConvergence` instead. Logging is covered more fully in the logging documentation; the `Convergence` logger itself requires no additional configuration beyond construction.

## Implicit vs explicit residual norms

The two norm-based criteria reflect a classical accuracy-vs-cost trade-off:

- **Explicit** (`ResidualNorm`): computes `‖b − A x‖` directly with a SpMV each time the criterion is checked. Accurate — the norm matches what the user actually cares about — but at the cost of **one additional SpMV per iteration** on top of the SpMV that the solver would otherwise perform for its own iteration step. For matrix-bandwidth-bound solvers (which most are), that roughly doubles the per-iteration cost.
- **Implicit** (`ImplicitResidualNorm`): reads the residual estimate that the Krylov recurrence maintains for free as a by-product of the iteration. No extra SpMV, but the estimate can drift away from the true residual in finite-precision arithmetic after many iterations, particularly when the solver is near the noise floor of the working precision.

Prefer explicit residual norms in production runs where correctness is paramount. Use implicit norms in performance-sensitive inner loops (e.g., a preconditioner's inner solve), where a loose stopping tolerance already forgives some inaccuracy.

:::{seealso}
- [Solvers — taxonomy](solvers.md) — what consumes stopping criteria.
- API reference: [`gko::stop`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/namespacegko_1_1stop.html)
:::
