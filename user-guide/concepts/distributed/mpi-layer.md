# The MPI layer

Ginkgo's distributed module sits on top of a thin C++ wrapper over MPI that lives in the
`gko::experimental::mpi` namespace (`include/ginkgo/core/base/mpi.hpp`). This page describes the
central abstractions — the communicator, the request, the collective communicator — and the
calls that the distributed `Matrix`, `Vector`, and preconditioners are built on. Most user code
does not invoke this layer directly; understanding it matters when implementing custom
distributed `LinOp` operations or debugging communication-related stalls.

## Build-time gate

The MPI layer is built when Ginkgo is configured with `-DGINKGO_BUILD_MPI=ON`. Without that flag,
the distributed `Matrix`, `Vector`, `RowGatherer`, the collective communicators and the distributed
preconditioners (and the corresponding sub-tree of the public API) belong to MPI-enabled builds only;
`Partition` and `index_map` are part of every build.

## The `mpi::environment` RAII guard

Ginkgo provides a small RAII helper for the canonical `MPI_Init` / `MPI_Finalize` pair:

```cpp
int main(int argc, char** argv) {
    gko::experimental::mpi::environment env{argc, argv};
    // ... distributed Ginkgo work ...
    return 0;   // env's destructor calls MPI_Finalize
}
```

The guard is for applications that have not called `MPI_Init` themselves. If your application
already initialises MPI, skip the guard and construct `mpi::communicator` objects directly;
`mpi::environment::is_initialized()` reports whether `MPI_Init` has run. The constructor takes an
optional `mpi::thread_type` (default `thread_type::serialized`, i.e. `MPI_THREAD_SERIALIZED`), and
`get_provided_thread_support()` returns the level MPI granted.

## The central `mpi::communicator`

`gko::experimental::mpi::communicator` is the central handle that the distributed `Matrix`, `Vector`
and `RowGatherer` carry. It wraps an `MPI_Comm` with shared ownership and exposes a method-style interface for the
collective operations Ginkgo needs:

```cpp
gko::experimental::mpi::communicator comm{MPI_COMM_WORLD};
int rank = comm.rank();
int size = comm.size();
MPI_Comm raw = comm.get();   // for direct MPI calls when you need them
```

The distributed `Matrix`, `Vector` and `RowGatherer` inherit from `DistributedBase` and expose
`get_communicator()` to return the same object:

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

By default Ginkgo stages device data through host memory for every MPI call. Configuring with
`-DGINKGO_FORCE_GPU_AWARE_MPI=ON` lets Ginkgo hand device pointers to MPI directly. The
communicator's `force_host_buffer` flag (set at construction) restores host staging for that
communicator in such a build — useful when one communicator spans a fabric without GPU support.
`mpi::requires_host_buffer(exec, comm)` combines the build setting, the executor and this flag; see
[Stream-MPI ordering](communication-primitives.md#stream-mpi-ordering-for-gpu-executors) in the
communication-primitives page.

## The `mpi::request` handle

Non-blocking collective operations return a `mpi::request` — a thin, move-only wrapper over
`MPI_Request`. It exposes a single completion call, `wait()`, plus `get()` for the underlying
`MPI_Request*`:

```cpp
auto req = comm.i_broadcast(exec, buf, count, root);
// ... do unrelated work in parallel ...
auto stat = req.wait();   // block until the broadcast completes; returns mpi::status
```

For a non-blocking completion check, call `MPI_Test` directly on the raw handle returned by
`req.get()`. The standard idiom for waiting on
many outstanding requests is the free function `mpi::wait_all(std::vector<request>&)`, which calls
`wait()` on each request in turn.

## Supported collective operations

The communicator exposes a near-complete subset of MPI's collective interface. The full surface
is documented in `mpi.hpp`; the table below lists where Ginkgo's own distributed code invokes it,
and the remaining operations are there for application code:

| Operation | Method on `communicator` | Used by |
|---|---|---|
| Allreduce | `all_reduce`, `i_all_reduce` | `Vector::compute_dot` / `compute_conj_dot` / `compute_norm1` / `compute_norm2` (blocking `all_reduce`), the GMRES CGS/CGS2 Hessenberg reduction |
| Allgather | `all_gather`, `i_all_gather` | the partition helpers (`build_partition_from_local_size`, `build_partition_from_local_range`) |
| All-to-all | `all_to_all`, `i_all_to_all` | send-size discovery in `DenseCommunicator` / `NeighborhoodCommunicator` construction, `assembly_mode::communicate` |
| **All-to-all-variable** | `all_to_all_v`, `i_all_to_all_v` | **the halo exchange in distributed SpMV**, `assembly_mode::communicate`, distributed SpGEMM, Pgm / UniformCoarsening setup |
| Broadcast, reduce, scan, gather, scatter (+ `_v`), send / recv | `broadcast`, `reduce`, `scan`, `gather`, `gather_v`, `scatter`, `scatter_v`, `send`, `recv` and the `i_` variants | application code |

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
  works on every MPI implementation; it is non-blocking except when Ginkgo is configured against
  Open MPI older than 4.1 (see [Dense communicator](dense-communicator.md)).
- **`NeighborhoodCommunicator`** — `MPI_Ineighbor_alltoallv` over a graph topology that reflects
  only the ranks this process actually communicates with. Asymptotically cheaper on
  sparse-communication patterns.

Both satisfy the same interface; the distributed `Matrix` doesn't care which one is used. See the
[Collective communicator](collective-communicator.md) page for the interface contract and the
two implementations.

## How distributed `Matrix::apply` uses the layer

Putting it together, a single distributed `apply` invokes the layer roughly as follows on the host
executors (on device executors the diagonal-matrix kernel is launched before the exchange is issued;
see [Communication primitives](communication-primitives.md)):

```text
1. RowGatherer packs the values to send into a contiguous buffer (host or device).
2. coll_comm->i_all_to_all_v(exec, send_ptr, recv_ptr) returns an mpi::request.
3. The diagonal-matrix kernel runs in parallel with the in-flight collective (with the blocking
   Open MPI < 4.1 fallback of DenseCommunicator the collective has already completed).
4. request.wait() — the recv buffer now holds x_halo.
5. The off-diagonal-matrix kernel applies to x_halo and accumulates into y_local.
```

This is the only path that performs MPI traffic during a normal solve. Reductions for norms,
dot products and stopping criteria use the blocking `all_reduce` on the same communicator. No point-to-point
`MPI_Isend` / `MPI_Irecv` traffic happens — the SpMV path is collective end-to-end.

:::{seealso}
- [Communication primitives](communication-primitives.md) — how the layer is consumed by the SpMV
  pipeline.
- [Collective communicator](collective-communicator.md) — the interface and its two
  implementations (Dense and Neighborhood).
- [RowGatherer](row-gather-scatter.md) — the helper that drives the halo
  exchange built on top of the collective communicator.
- [Distributed solvers and preconditioners](solvers-and-preconditioners.md) — where
  `all_reduce` shows up.
- API reference: [`gko::experimental::mpi`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/namespacegko_1_1experimental_1_1mpi.html), [`gko::experimental::mpi::CollectiveCommunicator`](../../../api-reference/mpi/collective-communicator.md)
:::
