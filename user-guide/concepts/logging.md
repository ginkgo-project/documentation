# Logging and observability

Ginkgo's `Logger` interface is the primary observability hook — it lets you watch what's happening inside a solver, profile kernels, or capture convergence data without touching solver code. This page covers the built-in loggers, where they hook in, and how to attach them.

## The Logger interface

A `Logger` is a callback receiver. The `LinOp` and `Executor` runtimes call into the logger at well-defined events:

- **LinOp lifecycle:** `on_linop_apply_started`, `on_linop_apply_completed`, `on_linop_advanced_apply_started`, and their `_completed` counterparts.
- **Iterative solver:** `on_iteration_complete` — fires once per iteration with iteration count, residual, and current solution.
- **Executor:** kernel launches, memory allocations, copies, and synchronization points.

You attach a logger to a `LinOp` (typically a solver) before calling `apply`:

```cpp
auto solver = factory->generate(matrix);
solver->add_logger(my_logger);
solver->apply(b, x);
```

Or attach it to an executor to observe every kernel and allocation on that device:

```cpp
exec->add_logger(my_logger);
```

A single logger instance can be attached to multiple objects simultaneously. Loggers are stored as `shared_ptr` — the owning object does not transfer ownership.

:::{note}
Logging is currently not MPI-aware. Each rank logs its own events independently. If you need cross-rank timing, coordinate outside of Ginkgo.
:::

## Built-in loggers

| Logger             | What it captures                                   | Use case                               |
| ------------------ | -------------------------------------------------- | -------------------------------------- |
| `Convergence`      | Iteration count and final residual norm            | After solve: did it converge well?     |
| `Stream`           | Logs every event to a stream (`cout`, `ofstream`)  | Tracing and debugging                  |
| `PerformanceHint`  | Repeated cross-executor copies and allocations     | Finding avoidable transfers            |
| `ProfilerHook`     | NVTX / ROCTX / VTune / TAU named regions, or a timing summary | Profiler integration and quick timings |
| `BatchConvergence` | Per-batch-item iteration count and residual        | Batched solvers                        |

All built-in loggers live in the `gko::log` namespace; `BatchConvergence` lives in `gko::batch::log`.

## Convergence — the most common use case

`Convergence` captures the final state of an iterative solve: the total iteration count and the residual norm at termination. It is the right first logger to add when you want to know whether a solve succeeded.

```cpp
auto convergence = gko::share(gko::log::Convergence<double>::create());
solver->add_logger(convergence);
solver->apply(b, x);

std::cout << "Converged in " << convergence->get_num_iterations() << " iterations\n";
auto resnorm = convergence->get_residual_norm();   // Dense<double> on `exec`, shape 1×nrhs
```

`Convergence` is templated on the value type and is created with its static `create()` function, which returns a `std::unique_ptr`; `gko::share` turns it into the `shared_ptr` that `add_logger` takes. The residual norm is a 1×nrhs `Dense` living on the solver's executor.

:::{attention}
`get_residual_norm()` returns a `LinOp*` to a `Dense` that lives on the **solver's
executor**. If the solver runs on a `CudaExecutor`, `HipExecutor`, or
`DpcppExecutor`, the underlying `get_const_values()` pointer is a *device* pointer
— dereferencing it on the host (e.g., reading `data[0]` directly to print a value)
is undefined behaviour and typically segfaults.

To read the values on the host, clone the residual-norm vector to a host-resident
executor first:

```cpp
auto host = gko::ReferenceExecutor::create();
auto resnorm_host = gko::clone(host,
    gko::as<gko::matrix::Dense<double>>(convergence->get_residual_norm()));
std::cout << "norm = " << resnorm_host->at(0, 0) << "\n";
```

The same warning applies to `get_implicit_sq_resnorm()`.
:::

The logger reads the outcome from the [stopping criteria](stopping-criteria.md) that triggered termination, so what counts as "converged" is fully controlled by the criteria you configure on the solver.

## Stream logger — tracing

`Stream` logs every observable event to an `std::ostream`. It is the easiest way to see exactly what a solver is doing:

```cpp
auto stream_logger = gko::share(
    gko::log::Stream<double>::create(gko::log::Logger::all_events_mask, std::cout, true));
solver->add_logger(stream_logger);
solver->apply(b, x);
```

The third argument of `create(mask, os, verbose)` controls verbosity — set it to `true` to include matrix and vector values in the output, or `false` for event names only. Stream logging is most useful during the first integration of a new solver or when tracking down an unexpected convergence failure.

:::{attention}
Stream logging is synchronous and can be noisy. Disable it in production paths — every event incurs a format-and-write call.
:::

## ProfilerHook — integrating with external profilers

`ProfilerHook` emits named regions that profiling tools understand. The convenience
factory `create_for_executor` picks the appropriate backend for the active
executor:

| Factory | Backend | Used by |
|---|---|---|
| `create_nvtx(color_argb = …)` | NVTX | NSight Systems on CUDA |
| `create_roctx()` | ROCTX | rocprof on HIP |
| `create_vtune()` | Intel VTune ITT | VTune on CPU and DPC++ |
| `create_tau(initialize = true)` | TAU via PerfStubs | TAU and other PerfStubs targets |
| `create_for_executor(exec)` | Picks the right one of the above for the executor's backend (NVTX for CUDA, ROCTX for HIP on AMD, VTune for DPC++, TAU otherwise) | General-purpose default |
| `create_summary()` | Accumulates wall-clock time per named range and prints a table when the logger is destroyed | Quick timing without an external tool |
| `create_custom(begin, end)` | User-supplied callbacks | Any profiler with a C entry point — including **Caliper** |

```cpp
auto profiler = gko::log::ProfilerHook::create_for_executor(exec);
exec->add_logger(profiler);   // before generate, to include solver setup
```

Solvers, factories and stopping criteria on that executor forward their events to its propagating
loggers, so one attachment covers apply, generate, criterion checks, kernels, allocations and copies.
In the profiler UI you will see solver phases grouped by region name, which makes
it straightforward to identify which part of a solve is expensive.

### Caliper

Caliper is reached through the generic custom hook, `create_custom`. Pass two function objects that call Caliper's C API directly
(`cali_begin_region` / `cali_end_region`) as the begin/end callbacks, and Ginkgo
will invoke them at every annotated region boundary.

A worked example of Caliper integration in an application is in
[How-To: ProfilerHook with Caliper](../how-to/using/profiler-hook-caliper.md).

## Performance logger

Two loggers report on performance without an external profiler: a timing summary and a hint logger.

`ProfilerHook::create_summary()` accumulates wall-clock durations per named range, without emitting external profiling markers, and prints a table when the logger is destroyed. The default timer measures on the CPU; passing `gko::Timer::create_for_executor(exec)` (or calling `set_synchronization(true)` on the logger) makes the GPU timings reliable:

```cpp
auto summary = gko::log::ProfilerHook::create_summary(
    gko::Timer::create_for_executor(exec));
exec->add_logger(summary);
// ... run workload ...
```

`PerformanceHint` analyses allocations and copies on an executor and reports unnecessary ones. It checks for repeated cross-executor copies from or to the same pointer and for repeated allocation/free pairs of the same size:

```cpp
auto hint = gko::share(gko::log::PerformanceHint::create(std::cerr));
exec->add_logger(hint);
// ... run workload ...
hint->print_status();   // writes the hints collected so far
```

Both attach to an executor rather than a solver, so they capture all operations on that device, not just the ones inside a single `apply` call. This is useful for characterising the total time a benchmark spends in Ginkgo.

## Batched: BatchConvergence

[Batched solvers](batched.md) solve many independent systems simultaneously. Each batch item has its own iteration count and residual, so the scalar `Convergence` logger does not apply. Use `gko::batch::log::BatchConvergence` instead:

```cpp
auto bcv = gko::share(gko::batch::log::BatchConvergence<double>::create());
batched_solver->add_logger(bcv);
batched_solver->apply(rhs, sol);

auto iters = bcv->get_num_iterations();   // array<int>: per-item iteration count
auto norms = bcv->get_residual_norm();    // array<double>: per-item final residual norm
```

Both returned arrays have one entry per batch item and live on the solver's executor. Copy to host to inspect individual values.

## Custom loggers

If the built-in loggers do not fit your use case, subclass `gko::log::Logger` and override exactly the events you care about:

```cpp
class MyLogger : public gko::log::Logger {
public:
    void on_iteration_complete(
        const gko::LinOp* solver,
        const gko::LinOp* b,
        const gko::LinOp* x,
        const gko::size_type& num_iterations,
        const gko::LinOp* residual,
        const gko::LinOp* residual_norm,
        const gko::LinOp* implicit_sq_residual_norm,
        const gko::array<gko::stopping_status>* status,
        bool all_stopped) const override
    {
        // capture what you need
    }
};
```

The key rule: do not block inside a logger. Loggers run in the hot path of `apply` — heavy I/O or synchronisation here will dominate your solve time.

## Publications

- [Towards Continuous Benchmarking: An Automated Performance Evaluation Framework for High Performance Software](https://doi.org/10.1145/3324989.3325719)

:::{seealso}
- [Stopping criteria](stopping-criteria.md) — `Convergence` reports the criterion's outcome.
- [The Executor model](executor-model.md) — loggers attach to executors too.
- [Batched solvers](batched.md) — `BatchConvergence` is the analogue for batched workloads.
- API reference: [`gko::log`](../../api-reference/log/index.md)
:::
