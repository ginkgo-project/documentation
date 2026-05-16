# Reference executor

`gko::ReferenceExecutor` is a single-threaded host executor. It is the correctness baseline for every other backend — kernels are written first in `reference/`, and other backends are validated against it.

## Construction

```cpp
auto host = gko::ReferenceExecutor::create();
```

No arguments. Always available, regardless of build configuration.

## When to use it

- **Unit tests.** Deterministic results — no parallel non-determinism, no race conditions.
- **Debugging numerical issues.** Step through kernels with `gdb` or a similar tool.
- **CI smoke tests.** Verify correctness in environments without GPUs.

## When not to use it

For performance. `ReferenceExecutor` makes no attempt to parallelize. For host-side performance use `OmpExecutor`.

## Memory model

Plain host memory via `new[]`/`delete[]` (`CpuAllocator`). Memory allocated by a `ReferenceExecutor` is also accessible from any `OmpExecutor` instance — `ReferenceExecutor::memory_accessible(omp_exec)` is `true`, since both share the same host address space.

:::{seealso}
- [The Executor model](../executor-model.md)
- API reference: [`gko::ReferenceExecutor`](../../../api-reference/executors/reference.md)
:::
