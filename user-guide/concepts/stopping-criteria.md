# Stopping criteria

An iterative solver needs a way to decide when to stop. Ginkgo's stopping criteria are composable factories that solvers consult each iteration. This page covers the built-in criteria, how they compose, and how to plug them into solver factories.

## The abstraction

A stopping criterion factory produces a runtime criterion object that the solver calls after each iteration. The criterion sees the current iteration count, the current residual, and the current solution vector; it returns a per-right-hand-side converged flag. When all right-hand sides are flagged, the solver stops.

Because criteria are factories (the same pattern as solvers and preconditioners), they plug into solver factories via `with_criteria(...)` and are constructed together with the solver at `generate` time. This keeps criterion configuration independent of the solver algorithm.

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

- **`Iteration`**: unconditionally stops after `max_iters` iterations. A safety net that prevents infinite loops regardless of convergence.
- **`ResidualNorm`**: stops when `‖r‖ / baseline ≤ reduction_factor`. The `baseline` can be `initial_resnorm` (relative to the initial residual), `rhs_norm` (relative to `‖b‖`), or `absolute` (no normalisation). The norm comes from the solver's residual norm or residual vector (for example CG, FCG, PipeCG, BiCG, BiCGSTAB, CGS, GMRES, CB-GMRES, GCR and IDR pass one of them; IR and Chebyshev pass the explicitly computed `b − A x`), so no extra SpMV is needed. Solvers that pass neither, such as MINRES and Multigrid, let the criterion compute `r = b − A x` with an SpMV each time it is checked.
- **`ImplicitResidualNorm`**: uses the squared residual estimate that the solver maintains inside its recurrence and passes to the criterion (CG, FCG, PipeCG, BiCG, BiCGSTAB, CGS and MINRES). No extra SpMV, but the estimate can drift from the true residual over many iterations. See the *Implicit vs explicit residual norms* section below for the trade-off.
- **`Time`**: stops after a wall-clock duration regardless of convergence. Useful for anytime algorithms or hard latency budgets.

## Composition with `Combined`

Multiple criteria compose via `gko::stop::Combined`, which always stops as soon as *any* sub-criterion fires (OR semantics). Pass the sub-criteria as factories:

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

This stops when either 1000 iterations have run *or* the residual has reduced by a factor of 1e-8, whichever comes first. Using at least an `Iteration` bound together with a `ResidualNorm` criterion is good practice: the iteration bound guards against divergence, while the norm criterion avoids unnecessary extra work after convergence.

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

A factory returned by `.on(exec)` is a `std::unique_ptr`, so pass an existing `Combined` factory with `std::move` (or wrap it with `gko::share`):

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

For batched solvers, use `gko::batch::log::BatchConvergence` instead. Logging is covered more fully in the logging documentation; the `Convergence` logger itself requires no additional configuration beyond construction.

## Implicit vs explicit residual norms

The two norm-based criteria reflect a classical accuracy-vs-cost trade-off:

- **Explicit** (`ResidualNorm`): measures the norm of the residual that the solver hands over (the recurrence residual for the CG family, the Givens-rotation estimate for GMRES, the explicitly computed `b − A x` for IR and Chebyshev). The check costs one norm computation. Solvers that pass neither a residual nor its norm (MINRES, Multigrid) make the criterion evaluate `b − A x`, which adds **one SpMV per check** on top of the solver's own SpMV and, for matrix-bandwidth-bound solvers, can roughly double the per-iteration cost. The recurrence-based residuals can drift from the true `‖b − A x‖` in finite precision.
- **Implicit** (`ImplicitResidualNorm`): reads the squared residual estimate that the solver's recurrence maintains for free as a by-product of the iteration (for CG-type solvers the preconditioned inner product $\langle r, z \rangle$ with $z = M^{-1} r$). No extra SpMV, and the global reduction for the norm is saved. With a preconditioner this is an $M^{-1}$-weighted quantity, a different measure from `‖b − A x‖`; in finite-precision arithmetic the estimate can also drift away from the true residual after many iterations, particularly near the noise floor of the working precision.

Prefer explicit residual norms in production runs where correctness is paramount. Use implicit norms in performance-sensitive inner loops (e.g., a preconditioner's inner solve), where a loose stopping tolerance already forgives some inaccuracy.

:::{seealso}
- [Solver taxonomy](solvers.md): what consumes stopping criteria.
- API reference: [`gko::stop`](../../api-reference/stop/index.md)
:::
