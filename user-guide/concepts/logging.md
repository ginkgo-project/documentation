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
| `Performance`      | Operation timings and allocation totals            | Lightweight profiling                  |
| `ProfilerHook`     | NVTX / ROCTX / VTune named regions                | GPU profiler integration               |
| `BatchConvergence` | Per-batch-item iteration count and residual        | Batched solvers                        |

All built-in loggers live in the `gko::log` namespace.

## Convergence — the most common use case

`Convergence` captures the final state of an iterative solve: the total iteration count and the residual norm at termination. It is the right first logger to add when you want to know whether a solve succeeded.

```cpp
auto convergence = std::make_shared<gko::log::Convergence<double>>();
solver->add_logger(convergence);
solver->apply(b, x);

std::cout << "Converged in " << convergence->get_num_iterations() << " iterations\n";
auto resnorm = convergence->get_residual_norm();   // Dense<double> on `exec`, shape 1×nrhs
```

`Convergence` is templated on the value type. The residual norm is a 1×nrhs `Dense` living on the solver's executor — copy it to a `OmpExecutor` or `ReferenceExecutor` to read individual values on the host.

The logger reads the outcome from the [stopping criteria](stopping-criteria.md) that triggered termination, so what counts as "converged" is fully controlled by the criteria you configure on the solver.

## Stream logger — tracing

`Stream` logs every observable event to an `std::ostream`. It is the easiest way to see exactly what a solver is doing:

```cpp
auto stream_logger = std::make_shared<gko::log::Stream<>>(std::cout, true);
solver->add_logger(stream_logger);
solver->apply(b, x);
```

The second constructor argument controls verbosity — set it to `true` to include matrix and vector values in the output, or `false` for event names only. Stream logging is most useful during the first integration of a new solver or when tracking down an unexpected convergence failure.

:::{attention}
Stream logging is synchronous and can be noisy. Disable it in production paths — every event incurs a format-and-write call.
:::

## ProfilerHook — integrating with nvprof / rocprof

`ProfilerHook` emits named regions that profiling tools understand. On CUDA backends it uses NVTX; on HIP backends it uses ROCTX; on CPU backends it hooks into Intel VTune ITT.

```cpp
auto profiler = gko::log::ProfilerHook::create_for_executor(exec);
solver->add_logger(profiler);
```

`create_for_executor` selects the right backend automatically based on the executor type. In the profiler UI (Nsight Systems, rocprof, VTune) you will see solver phases grouped by region name, which makes it straightforward to identify which part of a solve is expensive.

## Performance logger

`Performance` is lighter-weight than `ProfilerHook`. It accumulates wall-clock durations per operation type and per executor, without emitting external profiling markers:

```cpp
auto perf = std::make_shared<gko::log::Performance>();
exec->add_logger(perf);
// ... run workload ...
auto stats = perf->get_total_time(/* operation */);
```

Because it attaches to an executor rather than a solver, it captures all operations on that device, not just the ones inside a single `apply` call. This is useful for characterising the total GPU time a benchmark spends in Ginkgo.

## Batched: BatchConvergence

[Batched solvers](batched.md) solve many independent systems simultaneously. Each batch item has its own iteration count and residual, so the scalar `Convergence` logger does not apply. Use `BatchConvergence` instead:

```cpp
auto bcv = std::make_shared<gko::log::BatchConvergence<double>>();
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

:::{seealso}
- [Stopping criteria](stopping-criteria.md) — `Convergence` reports the criterion's outcome.
- [The Executor model](executor-model.md) — loggers attach to executors too.
- [Batched solvers](batched.md) — `BatchConvergence` is the analogue for batched workloads.
- API reference: [`gko::log`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/namespacegko_1_1log.html)
:::
