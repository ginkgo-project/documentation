# Switch executor

You have a working Ginkgo program on one executor and want to run the same
program on a different one, typically CPU → GPU. In Ginkgo, the executor
is a runtime argument: switch the constructor and everything downstream
follows.

## The change

The only line that differs between backends is the executor constructor.
The rest of the program (matrix construction, solver factory, `apply`)
is identical.

```cpp
// CPU, single-threaded reference baseline:
auto exec = gko::ReferenceExecutor::create();

// CPU, OpenMP-parallel:
auto exec = gko::OmpExecutor::create();

// NVIDIA GPU (CUDA). Pair with a host executor for staging.
auto exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create());

// AMD GPU (HIP):
auto exec = gko::HipExecutor::create(0, gko::OmpExecutor::create());

// Intel GPU (SYCL):
auto exec = gko::DpcppExecutor::create(0, gko::OmpExecutor::create());
```

The first argument to GPU executors is the device id; the second is the
**host executor** that the GPU executor uses for host-side allocations
and copy-staging. Pair the GPU executor with `OmpExecutor` (or
`ReferenceExecutor` if OpenMP is not built).

## Picking the device

For multi-GPU nodes, query the count and pick by rank or environment:

```cpp
int n = gko::CudaExecutor::get_num_devices();   // or HipExecutor / DpcppExecutor
auto exec = gko::CudaExecutor::create(rank % n, gko::OmpExecutor::create());
```

For MPI ranks on multi-GPU nodes, the helper
`gko::experimental::mpi::map_rank_to_device_id(comm, n)` gives each rank a
device id based on its node-local rank.

## What happens to existing data

Switching the executor in `main()` is enough: every `LinOp`, `Dense`,
and `array` constructed afterwards lives on the new executor. If you
already have host-resident matrices and want to copy them across, use
`gko::clone(target_exec, obj)` or the type's own `clone(target_exec)`:

```cpp
auto A_host = gko::matrix::Csr<>::create(host_exec);
A_host->read(host_matrix_data);

auto A_dev = gko::clone(exec, A_host);   // shallow type, deep data
```

For zero-copy from application-owned buffers see
[Zero-copy from application memory](zero-copy-application-memory.md).

## Common pitfalls

- **Operands on different executors.** `apply(b, x)` requires `b`, `x`,
  and the operator to share the same executor. If they don't, the call
  silently clones to align them, which is correct but copies data each
  call. Construct everything on the target executor from the start.
- **`at(i, j)` is host-only.** Dereferencing `Dense::at` on a vector
  whose executor is a GPU is undefined behaviour. `clone` to the host
  executor first when you need to read or write individual entries.
- **Share the host master across GPU executors.** Each `OmpExecutor`
  allocates its own thread pool at construction. If you build several
  GPU executors (e.g. one per MPI rank's local device), construct **one**
  `OmpExecutor` and pass the same `shared_ptr` to each GPU executor
  rather than calling `gko::OmpExecutor::create()` inline at every
  GPU-executor construction site.

:::{seealso}
- [Move data between executors](move-data-between-executors.md): `clone` vs `copy_from`.
- [Executor model](../../concepts/executor-model.md): the conceptual reference.
:::
