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

std::cout << "iters       = " << convergence->get_num_iterations()  << '\n';
std::cout << "converged   = " << convergence->has_converged()       << '\n';

// The residual norm is a Dense pointer on the solver's executor.
auto resnorm = gko::as<gko::matrix::Dense<double>>(
    convergence->get_residual_norm());
auto on_host = gko::clone(exec->get_master(), resnorm);
std::cout << "value       = " << on_host->at(0, 0) << '\n';
```

:::{attention}
`get_residual_norm()` returns a `Dense` whose storage lives on the
solver's executor. For a `CudaExecutor` / `HipExecutor` /
`DpcppExecutor` solver this is a **device** pointer: dereferencing it
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
// Log every event to std::cout.
auto stream = gko::share(gko::log::Stream<>::create(
    gko::log::Logger::all_events_mask, std::cout));
solver->add_logger(stream);
solver->apply(b, x);
```

You can narrow the event mask to focus on a subset (e.g. only
iteration completion and convergence):

```cpp
// Log only iteration completion and the stopping-criterion events.
auto mask = gko::log::Logger::iteration_complete_mask
          | gko::log::Logger::criterion_events_mask;
auto stream_logger = gko::share(gko::log::Stream<>::create(mask, std::cout));
solver->add_logger(stream_logger);
```

## Per-iteration residual history

`log::Record` keeps a structured record of every logged event in an
in-memory data structure, which you can iterate after the solve:

```cpp
auto record = gko::share(gko::log::Record::create(
    gko::log::Logger::all_events_mask, /*max_storage=*/0));
solver->add_logger(record);
solver->apply(b, x);

for (const auto& it : record->get().iteration_completed) {
    // it->num_iterations, it->implicit_sq_residual_norm, etc.
}
```

`max_storage = 0` keeps every event; the default of 1 keeps only the
most recent one of each kind.

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
`create_summary` (a per-operation timing table without an external
profiler), `create_for_executor` (auto-pick based on executor), and
`create_custom` (provide your own begin/end callbacks; this is the standard
path for Caliper, see
[Integrate ProfilerHook with Caliper](profiler-hook-caliper.md)).

## Detach a logger

```cpp
solver->remove_logger(convergence);
```

Loggers can be attached and detached at runtime, useful when you
want logging only during a setup phase.

## Common pitfalls

- **Logger lifetime.** `add_logger` takes a `std::shared_ptr`, so the
  attached object keeps the logger alive. Keep your own `gko::share`d
  pointer to read its results after the solve.
- **Add the logger before calling `apply`.** Events fire during
  `apply`; a logger attached after is silent for that call.
- **The `Convergence` residual is the *final* one, not per-iteration.**
  For per-iteration history use `Record` (in-memory) or `Stream`
  (to a stream).

:::{seealso}
- [Logging and observability](../../concepts/logging.md): the conceptual reference, including the full backend table for `ProfilerHook`.
- [Configure stopping criteria](configure-stopping-criteria.md).
:::
