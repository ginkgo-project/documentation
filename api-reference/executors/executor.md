# `gko::Executor`

Abstract base for every Ginkgo executor. Defines the interface every
concrete backend implements: kernel dispatch (`run`), allocation
(`alloc` / `free`), cross-executor copies (`copy_from`, `copy_val_to_host`),
synchronisation (`synchronize`), and the master-executor pairing that lets
device executors transparently transfer data through a paired host
executor. Application code generally holds a `std::shared_ptr<const
Executor>` and never names the concrete subclass — that's the point of the
abstraction.

```{doxygenclass} gko::Executor
:members:
```
