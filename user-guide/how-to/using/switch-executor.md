# Switch executor

:::{note}
Stub — content forthcoming.
:::

Move a working program from one executor (`ReferenceExecutor` / `OmpExecutor` /
`CudaExecutor` / `HipExecutor` / `DpcppExecutor`) to another with minimal code
changes. Covers `Executor::create`, the master/slave pairing for GPU
executors, and what `clone(exec)` does to existing `LinOp`s and arrays.
