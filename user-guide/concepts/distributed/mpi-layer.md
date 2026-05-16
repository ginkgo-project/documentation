# The MPI layer

Ginkgo's distributed module sits on top of a thin C++ wrapper over MPI that lives in the
`gko::experimental::mpi` namespace (`include/ginkgo/core/base/mpi.hpp`). This page describes the
central abstractions — the communicator, the request, the collective communicator — and the
calls that the distributed `Matrix`, `Vector`, and preconditioners are built on. Most user code
does not invoke this layer directly; understanding it matters when implementing custom
distributed `LinOp` operations or debugging communication-related stalls.

## Build-time gate

The MPI layer is built when Ginkgo is configured with `-DGINKGO_BUILD_MPI=ON`. Without that flag,
the `gko::experimental::distributed::*` types are not available and the corresponding sub-tree of
the public API is not installed.

## The `mpi::environment` RAII guard

Ginkgo provides a small RAII helper for the canonical `MPI_Init` / `MPI_Finalize` pair:

```cpp
int main(int argc, char** argv) {
    gko::experimental::mpi::environment env{argc, argv};
    // ... distributed Ginkgo work ...
    return 0;   // env's destructor calls MPI_Finalize
}
```

If your application already initialises MPI elsewhere, you can skip the guard — Ginkgo detects
that initialisation via `MPI_Initialized` and reuses the existing environment. The constructor
also lets you request a thread-support level (defaults to `MPI_THREAD_SERIALIZED`).

## The central `mpi::communicator`

`gko::experimental::mpi::communicator` is the central handle that every distributed Ginkgo object
carries. It wraps an `MPI_Comm` with shared ownership and exposes a method-style interface for the
collective operations Ginkgo needs:

```cpp
gko::experimental::mpi::communicator comm{MPI_COMM_WORLD};
int rank = comm.rank();
int size = comm.size();
MPI_Comm raw = comm.get();   // for direct MPI calls when you need them
```

Every distributed object inherits from `DistributedBase` and exposes `get_communicator()` to
return the same object:

```cpp
auto comm = dist_matrix->get_communicator();   // same type as the one constructed above
```

### Construction and ownership

Three construction patterns are supported:

| Constructor | Ownership | Use when |
|---|---|---|
| `communicator(MPI_Comm)` | non-owning | You already manage the underlying `MPI_Comm` lifetime (typically `MPI_COMM_WORLD` or a comm split managed by your application). |
| `communicator(MPI_Comm, color, key)` / `communicator(communicator, color, key)` | shared (owning the result of the split) | You want a sub-communicator. Internally calls `MPI_Comm_split`; the resulting `MPI_Comm` is freed when the last copy of the wrapper is destroyed. |
| `communicator::create_owning(MPI_Comm)` | shared (owning the supplied `MPI_Comm`) | You want Ginkgo to take ownership of an already-created `MPI_Comm`. |

The wrapper is value-semantic and cheap to copy — copies share ownership of the underlying handle
through a `std::shared_ptr`. Pass it by value freely; you do not need to wrap it in a
`shared_ptr` again.

### Host-buffer override

The communicator has a `force_host_buffer` flag (set at construction). When `true`, every
collective operation goes through host memory regardless of whether the executor is on the
device. This is the safety hatch for systems whose MPI implementation is not CUDA/HIP-aware.
Ginkgo also detects this case automatically via the `mpi::requires_host_buffer(exec, comm)`
helper — see [Stream-MPI ordering](communication-primitives.md#stream-mpi-ordering-for-gpu-executors)
in the communication-primitives page.

## The `mpi::request` handle

Non-blocking collective operations return a `mpi::request` — a thin, move-only wrapper over
`MPI_Request`. It exposes a single completion call, `wait()`, plus `get()` for the underlying
`MPI_Request*`:

```cpp
auto req = comm.i_broadcast(exec, buf, count, root);
// ... do unrelated work in parallel ...
auto stat = req.wait();   // block until the broadcast completes; returns mpi::status
```

`request` does not provide a `test()` member. If you need a non-blocking completion check, call
`MPI_Test` directly on the raw handle returned by `req.get()`. The standard idiom for waiting on
many outstanding requests is the free function `mpi::wait_all(std::vector<request>&)`, which calls
`wait()` on each request in turn.

## Supported collective operations

The communicator exposes a near-complete subset of MPI's collective interface. The full surface
is documented in `mpi.hpp`; the operations Ginkgo's distributed types actually invoke are listed
below:

| Operation | Method on `communicator` | Used by |
|---|---|---|
| Broadcast | `broadcast`, `i_broadcast` | matrix construction (broadcasting partition info), debug helpers |
| Allreduce | `all_reduce`, `i_all_reduce` | distributed `Vector::compute_norm`, `compute_dot`, etc. |
| Reduce / scan | `reduce`, `i_reduce`, `scan`, `i_scan` | partition-helper algorithms |
| Gather / Allgather | `gather`, `i_gather`, `gather_v`, `all_gather`, `i_all_gather` | partition setup, debugging |
| All-to-all | `all_to_all`, `i_all_to_all` | dense communicator paths |
| **All-to-all-variable** | `all_to_all_v`, `i_all_to_all_v` | **the halo exchange in distributed SpMV** |
| Send / Recv | `send`, `recv`, `i_send`, `i_recv` | low-level building blocks; not used by the SpMV path |

The naming convention is consistent: methods without an `i_` prefix block, methods prefixed with
`i_` return an `mpi::request` and the caller is responsible for `wait()`-ing on it.

### The collective communicator

Distributed matrices do not invoke `i_all_to_all_v` directly. They go through a
`mpi::CollectiveCommunicator` (`include/ginkgo/core/distributed/collective_communicator.hpp`),
which is a higher-level object that knows the per-rank send and receive sizes and exposes:

```cpp
auto req = coll_comm->i_all_to_all_v(exec, send_ptr, recv_ptr);
```

Two implementations ship with Ginkgo:

- **`DenseCommunicator`** — non-blocking variable-size all-to-all (`MPI_Ialltoallv`) over the
  full rank set, with the per-rank size vector dictating who sends what to whom. The default,
  works on every MPI implementation.
- **`NeighborhoodCommunicator`** — `MPI_Ineighbor_alltoallv` over a graph topology that reflects
  only the ranks this process actually communicates with. Asymptotically cheaper on
  sparse-communication patterns.

Both satisfy the same interface; the distributed `Matrix` doesn't care which one is used. See the
[Collective communicator](collective-communicator.md) page for the interface contract and the
two implementations.

## How distributed `Matrix::apply` uses the layer

Putting it together, a single distributed `apply` invokes the layer roughly as follows:

```text
1. RowGatherer packs the values to send into a contiguous buffer (host or device).
2. coll_comm->i_all_to_all_v(exec, send_ptr, recv_ptr) returns an mpi::request.
3. The diagonal-matrix kernel runs in parallel with the in-flight collective.
4. request.wait() — the recv buffer now holds x_halo.
5. The off-diagonal-matrix kernel applies to x_halo and accumulates into y_local.
```

This is the only path that performs MPI traffic during a normal solve. Reductions for stopping
criteria and norms go through `i_all_reduce` on the same communicator. No point-to-point
`MPI_Isend` / `MPI_Irecv` traffic happens — the SpMV path is collective end-to-end.

:::{seealso}
- [Communication primitives](communication-primitives.md) — how the layer is consumed by the SpMV
  pipeline.
- [Collective communicator](collective-communicator.md) — the interface and its two
  implementations (Dense and Neighborhood).
- [RowGatherer and RowScatterer](row-gather-scatter.md) — the helper that drives the halo
  exchange built on top of the collective communicator.
- [Distributed solvers and preconditioners](solvers-and-preconditioners.md) — where
  `i_all_reduce` shows up.
- API reference: [`gko::experimental::mpi`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/namespacegko_1_1experimental_1_1mpi.html), [`gko::experimental::mpi::CollectiveCommunicator`](../../../api-reference/mpi/collective-communicator.md)
:::
