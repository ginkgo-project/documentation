# Use an existing MPI communicator

When your application has already created an `MPI_Comm`, typically a
sub-communicator from `MPI_Comm_split`, a duplicated comm from
`MPI_Comm_dup`, or one inherited from a framework like PETSc, wrap
that handle in `gko::experimental::mpi::communicator` and Ginkgo will
use it for every distributed object you build with it.

## The recipe

```cpp
#include <ginkgo/ginkgo.hpp>
#include <mpi.h>

// Your application's existing communicator is app_comm.
// Wrap it. The wrapper is non-owning — the caller still controls the lifetime
// of app_comm (i.e. when MPI_Comm_free is called).
auto gko_comm = gko::experimental::mpi::communicator{app_comm};

// Now every distributed type you build with gko_comm uses your handle.
auto partition = gko::share(
    gko::experimental::distributed::Partition<int, long>::
        build_from_global_size_uniform(exec, gko_comm.size(), global_n));

auto A = gko::experimental::distributed::Matrix<double, int, long>::create(
    exec, gko_comm);
A->read_distributed(data, partition);
```

## When to take ownership

The default constructor is **non-owning**: Ginkgo will not call
`MPI_Comm_free` on the handle. If you want Ginkgo to manage the
lifetime (rare; typically applications track their own comms), use
the explicit factory:

```cpp
auto owning = gko::experimental::mpi::communicator::create_owning(app_comm);
// MPI_Comm_free called when the last copy goes out of scope.
```

## Splitting communicators

The wrapper also exposes `MPI_Comm_split` directly, useful when you
want to subdivide ranks for a nested solver without dropping into
raw MPI:

```cpp
auto world = gko::experimental::mpi::communicator{MPI_COMM_WORLD};
auto sub   = gko::experimental::mpi::communicator{world,
                                                  /*color=*/rank / 4,
                                                  /*key=*/rank};
```

This produces sub-communicators with 4 ranks each. Use them like any
other `gko::experimental::mpi::communicator`.

## Force host-buffer staging

Ginkgo stages the data of device executors through host memory by
default. A build configured with `-DGINKGO_FORCE_GPU_AWARE_MPI=ON` hands
device pointers to MPI directly. For a single communicator that should
still use host staging in such a build, set the second constructor
argument:

```cpp
auto gko_comm = gko::experimental::mpi::communicator{app_comm,
                                                     /*force_host_buffer=*/true};
```

Every distributed exchange of that communicator then goes through a host
staging buffer, independent of the executor.

## Pair with the right device id

On multi-GPU nodes, choose the device based on the rank within the
communicator:

```cpp
auto n_dev = gko::CudaExecutor::get_num_devices();
auto exec  = gko::CudaExecutor::create(
    gko::experimental::mpi::map_rank_to_device_id(app_comm, n_dev),
    gko::OmpExecutor::create());
```

`map_rank_to_device_id` ensures ranks sharing a node get distinct
devices.

## Common pitfalls

- **MPI must be initialised first.** Either call `MPI_Init` yourself
  before constructing any `communicator`, or use
  `gko::experimental::mpi::environment` (Ginkgo's RAII guard around
  `MPI_Init` / `MPI_Finalize`).
- **One `MPI_Comm` per Ginkgo communicator wrapper.** Copies of the
  wrapper share the same `MPI_Comm`; freeing the handle while any
  copy survives is undefined behaviour. Stick with non-owning
  semantics unless you have a clear reason otherwise.
- **Distributed Ginkgo requires `-DGINKGO_BUILD_MPI=ON`.** The `mpi::`
  wrappers and the distributed `Matrix`, `Vector`, `RowGatherer` and
  `Schwarz` are compiled only in MPI-enabled builds; `Partition` and
  `index_map` are part of every build. See
  [Build options](../../../getting-started/build-options.md).

:::{seealso}
- [MPI layer](../../concepts/distributed/mpi-layer.md): the conceptual reference for the wrapper, request, and collective layer.
- [Assemble a distributed matrix](assemble-distributed-matrix.md): using the communicator to build a `distributed::Matrix`.
:::
