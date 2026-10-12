# Log convergence and timings

Loggers attach to a solver (or a stopping criterion, or any `LinOp`)
and receive callbacks when interesting events happen. This page covers
the three loggers you reach for most: `Convergence`, `Stream`, and
`ProfilerHook`.

## Capture the final residual and iteration count

`log::Convergence` is the lightweight option: it stores the iteration
count and the residual norm at the **point of convergence** (or at
max-iter, whichever fires first).

```cpp
auto convergence = gko::share(gko::log::Convergence<double>::create());
solver->add_logger(convergence);

solver->apply(b, x);

std::cout << "converged   = " << convergence->has_converged()       << '\n';
std::cout << "iters       = " << convergence->get_num_iterations()  << '\n';

// The residual norm is a Dense pointer on the solver's executor.
auto resnorm = gko::as<gko::matrix::Dense<double>>(
    convergence->get_residual_norm());
auto on_host = gko::clone(exec->get_master(), resnorm);
std::cout << "value       = " << on_host->at(0, 0) << '\n';
```

:::{attention}
`get_residual_norm()` returns a `Dense` whose storage lives on the
solver's executor. For a `CudaExecutor` / `HipExecutor` /
`DpcppExecutor` solver this is a **device** pointer — dereferencing it
on the host segfaults. `clone` to the master executor first, as shown
above.
:::

`get_implicit_sq_resnorm()` returns the *squared* implicit residual
(when the solver computes one); the same device-pointer caveat
applies.

## Stream every event

`log::Stream` writes a textual record of every logged event to a
stream. Verbose, but great for first-time debugging:

```cpp
solver->add_logger(gko::log::Stream<double>::create(
    gko::log::Logger::all_events_mask, std::cout));
solver->apply(b, x);
```

You can narrow the event mask to focus on a subset (e.g. only
iteration completion and convergence):

```cpp
auto mask = gko::log::Logger::iteration_complete_mask
          | gko::log::Logger::criterion_events_mask;
auto stream_logger = gko::log::Stream<double>::create(mask, std::cout);
```

## Per-iteration residual history

`log::Record` keeps a structured record of the logged events in an
in-memory data structure, which you can iterate after the solve:

```cpp
auto record = gko::share(gko::log::Record::create(
    gko::log::Logger::iteration_complete_mask, /*max_storage=*/0));  // 0 = keep all
solver->add_logger(record);
solver->apply(b, x);

for (const auto& it : record->get().iteration_completed) {
    // it->num_iterations, it->residual_norm, it->implicit_sq_residual_norm
}
```

`max_storage` defaults to 1, which keeps only the latest event of each kind. Record stores
copies of cloneable objects (vectors, norms), and its `solver` entry is empty for solvers.

This is the right shape when you want to plot a residual history or
dump the trace as JSON.

## Time individual operations

`log::ProfilerHook` emits region annotations into an external profiler
(NVTX, ROCTX, VTune, TAU, or a custom backend). Attach it to the
executor, not the solver, so allocations and kernel launches are
covered:

```cpp
exec->add_logger(gko::log::ProfilerHook::create_nvtx());
solver->apply(b, x);   // NVTX ranges now appear in nsys / Nsight
```

Other factories: `create_roctx`, `create_vtune`, `create_tau`,
`create_for_executor` (auto-pick based on executor), and
`create_custom` (provide your own begin/end callbacks — the standard
path for Caliper, see
[Integrate ProfilerHook with Caliper](profiler-hook-caliper.md)).

## Detach a logger

```cpp
solver->remove_logger(convergence);
```

Loggers can be attached and detached at runtime — useful when you
want logging only during a setup phase.

## Common pitfalls

- **Keep a handle to the logger.** Loggers are shared pointers; hold on
  to the one you created so you can read its data after the solve.
- **Add the logger before calling `apply`.** Events fire during
  `apply`; a logger attached after is silent for that call.
- **The `Convergence` residual is the *final* one, not per-iteration.**
  For per-iteration history use `Record` (in-memory, with `max_storage` of `0`) or `Stream`
  (to a stream).

:::{seealso}
- [Logging and observability](../../concepts/logging.md) — the conceptual reference, including the full backend table for `ProfilerHook`.
- [Configure stopping criteria](configure-stopping-criteria.md).
:::
