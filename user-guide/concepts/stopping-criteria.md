# Stopping criteria

An iterative solver needs a way to decide when to stop. Ginkgo's stopping criteria are composable factories that solvers consult each iteration. This page covers the built-in criteria, how they compose, and how to plug them into solver factories.

## The abstraction

A stopping criterion factory produces a runtime criterion object that the solver calls after each iteration. The criterion sees the current iteration count, the current residual, and the current solution vector; it returns a per-right-hand-side converged flag. When all right-hand sides are flagged, the solver stops.

Because criteria are factories — the same pattern as solvers and preconditioners — they plug into solver factories via `with_criteria(...)` and are constructed together with the solver at `generate` time. This keeps criterion configuration independent of the solver algorithm.

## Built-in criteria

Ginkgo ships four primary criteria. The `::build()` form is shown first; the shorthand helpers below need less code:

```cpp
// Stop after a fixed number of iterations.
gko::stop::Iteration::build()
    .with_max_iters(1000u)
    .on(exec);

// Stop when the residual norm drops below a fraction of the baseline.
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
- **`ResidualNorm`** — stops when `‖r‖ ≤ reduction_factor × baseline`. The `baseline` can be `rhs_norm` (the default, relative to `‖b‖`), `initial_resnorm` (relative to the initial residual), or `absolute` (no normalisation). Each check uses the residual norm the solver already provides (GMRES, CB-GMRES, GCR and IDR do) or takes the 2-norm of the residual vector the solver maintains, which costs one global reduction per check. Solvers that pass neither (MINRES, Multigrid) make it compute $b - A x$, an extra SpMV per check.
- **`ImplicitResidualNorm`** — checks the squared-norm estimate the solver maintains as a by-product of its recurrence (for CG-type solvers $\rho = r^H z$, a preconditioned quantity), which saves that reduction. Solvers that provide the estimate are CG, FCG, BiCG, BiCGSTAB, CGS, MINRES and PipeCG. See the *Implicit vs explicit residual norms* section below for the trade-off.
- **`Time`** — stops after a wall-clock duration regardless of convergence. Useful for anytime algorithms or hard latency budgets.

### Shorthand helpers

The helper functions build the same criteria without an executor and without a value type. Pass them directly to `with_criteria(...)`; the residual-norm criteria take their value type from the solver's vectors when the solver is generated:

```cpp
auto solver_factory = gko::solver::Cg<double>::build()
                          .with_criteria(
                              gko::stop::max_iters(1000u),
                              gko::stop::relative_residual_norm(1e-8))
                          .on(exec);
```

| Helper | Equivalent |
|---|---|
| `max_iters(n)` | `Iteration` with `max_iters = n` |
| `time_limit(duration)` | `Time` with `time_limit = duration` (any `std::chrono` unit) |
| `relative_residual_norm(tol)` | `ResidualNorm`, baseline `rhs_norm` |
| `initial_residual_norm(tol)` | `ResidualNorm`, baseline `initial_resnorm` |
| `absolute_residual_norm(tol)` | `ResidualNorm`, baseline `absolute` |
| `relative_implicit_residual_norm(tol)`, `initial_implicit_residual_norm(tol)`, `absolute_implicit_residual_norm(tol)` | `ImplicitResidualNorm` with the matching baseline |

The JSON short form uses the keys `iteration`, `time` and the residual-norm helper names (`relative_residual_norm`, `initial_residual_norm`, `absolute_residual_norm` and their `_implicit_` variants), for example `"criteria": {"iteration": 1000, "relative_residual_norm": 1e-8}`; see [Configuration files](config-files.md).

### Enforcing a minimum number of iterations

`gko::stop::min_iters(count, criteria...)` wraps one or more criteria so that they are checked only after `count` iterations have finished. From then on it behaves like the wrapped criteria, and several wrapped criteria are combined with OR. Criteria passed next to the wrapper, such as `max_iters`, keep their own behaviour. The wrapper suits inner solves and smoothers that must do a fixed amount of work even when the residual already looks small:

```cpp
auto solver_factory =
    gko::solver::Gmres<double>::build()
        .with_criteria(
            gko::stop::min_iters(10u, gko::stop::relative_residual_norm(1e-10)),
            gko::stop::max_iters(1000u))
        .on(exec);
```

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

A `Combined` factory built as above is a `std::unique_ptr`, so move it into `with_criteria`, or wrap it with `gko::share` to reuse it:

```cpp
auto solver_factory = gko::solver::Cg<double>::build()
                          .with_criteria(std::move(criteria))
                          .on(exec);
```

## Inspecting why a solver stopped

Solvers expose convergence information through a logger. Attach `gko::log::Convergence` before calling `apply` to read the final iteration count and residual norm afterward:

```cpp
auto convergence = gko::share(gko::log::Convergence<double>::create());
solver->add_logger(convergence);
solver->apply(b, x);

auto iters = convergence->get_num_iterations();
auto res   = convergence->get_residual_norm();
```

For batched solvers, use `gko::batch::log::BatchConvergence` instead. Logging is covered more fully in the logging documentation; the `Convergence` logger is created with its static `create()` function and needs no further configuration.

## Implicit vs explicit residual norms

The two norm-based criteria reflect an accuracy-vs-cost trade-off:

- **`ResidualNorm`** measures the 2-norm of the solver's residual; its cost is one global reduction per check, plus an extra SpMV per check for solvers that pass neither a norm nor a residual (MINRES, Multigrid). Use it when the tolerance must refer to the unpreconditioned residual; checking $\|b - A x\|$ after the solve gives a guaranteed bound.
- **`ImplicitResidualNorm`** reuses an inner product the solver computes anyway, so the check is free. The estimate is updated by short recurrences and, with a preconditioner, measures a preconditioned norm, so it can drift from $\|b - A x\|$ on long runs in finite precision, particularly near the noise floor of the working precision.

Prefer `ResidualNorm` in production runs where correctness is paramount. Use `ImplicitResidualNorm` in performance-sensitive inner solves with loose tolerances, paired with an `Iteration` cap.

:::{seealso}
- [Solvers — taxonomy](solvers.md) — what consumes stopping criteria.
- API reference: [`gko::stop`](../../api-reference/stop/index.md)
:::
