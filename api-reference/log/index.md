# `gko::log`

Observability hooks: loggers that attach to operators, solvers, and
executors to surface iteration counts, residual norms, allocation
events, and kernel timings. For higher-level guidance on which logger
to use see the
[Logging concept page](../../user-guide/concepts/logging.md).

## Base interface

- [`Logger`](logger.md) — abstract base; every logger overrides one or
  more `on_<event>` callbacks.
- [`Loggable`](loggable.md) — mixin that gives an object the ability to
  carry loggers and broadcast events to them.
- [`EnableLogging`](enable-logging.md) — CRTP convenience for objects
  whose own type is the loggable target.

## Built-in loggers

- [`Convergence`](convergence.md) — records iteration count and final
  residual norm of a solver run; the cheapest way to observe a Krylov
  solve.
- [`Stream`](stream.md) — writes events to a `std::ostream` in a
  human-readable form; useful for diagnostics during development.
- [`Record`](record.md) — captures all events into in-memory `Data`
  structs for later inspection or assertions in tests.
- [`SolverProgress`](solver-progress.md) — records the
  solver's per-iteration scalars (as a table or CSV stream) and optionally
  its vectors (as files).
- [`ProfilerHook`](profiler-hook.md) — emits scoped ranges to NVTX /
  ROCTX / VTune / TAU, so external profilers can attribute time to
  Ginkgo operations.
- [`PerformanceHint`](performance-hint.md) — heuristic checks that flag
  repeated cross-executor copies from or to the same pointer and repeated
  allocation/free pairs of the same size at runtime.

```{toctree}
:hidden:
:maxdepth: 1

logger
loggable
enable-logging
convergence
stream
record
solver-progress
profiler-hook
performance-hint
```
