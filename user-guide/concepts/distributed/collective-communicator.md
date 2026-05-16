# Collective communicators

Distributed `Matrix`, `Vector`, and helper objects do not invoke MPI collective routines directly.
Instead they go through `gko::experimental::mpi::CollectiveCommunicator`, an abstract interface
that exposes a single non-blocking variable-size all-to-all and the bookkeeping (per-rank send and
receive sizes) needed to drive it. Two concrete implementations ship with Ginkgo, and Ginkgo's
default is selected without the caller having to know which one runs.

## Why a separate abstraction

A halo exchange is a deterministic, repeated communication pattern: rank $r$ always sends the same
local indices to the same set of neighbours, and always receives the same remote indices from the
same neighbours. Once that pattern is computed (from the matrix's `IndexMap`), every subsequent
`apply` reuses it. The `CollectiveCommunicator` is the object that owns the pattern — the per-rank
send sizes, send offsets, recv sizes, recv offsets — so that the data path of `apply` is just:

```cpp
auto req = coll_comm->i_all_to_all_v(exec, send_buffer, recv_buffer);
req.wait();
```

The cost of computing the pattern is paid once at construction time (or whenever the matrix's
sparsity changes); the cost of the exchange itself is just the all-to-all.

## The interface

```cpp
class CollectiveCommunicator {
public:
    explicit CollectiveCommunicator(communicator base = MPI_COMM_NULL);

    const communicator& get_base_communicator() const;

    // Number of values this rank receives from / sends to its neighbours per exchange.
    virtual comm_index_type get_recv_size() const = 0;
    virtual comm_index_type get_send_size() const = 0;

    // The non-blocking variable-size all-to-all (typed and untyped overloads).
    template <typename SendType, typename RecvType>
    request i_all_to_all_v(std::shared_ptr<const Executor> exec,
                           const SendType* send_buffer,
                           RecvType* recv_buffer) const;

    request i_all_to_all_v(std::shared_ptr<const Executor> exec,
                           const void* send_buffer, MPI_Datatype send_type,
                           void* recv_buffer, MPI_Datatype recv_type) const;

    // Construct another communicator with the same dynamic type from a new base + index map.
    virtual std::unique_ptr<CollectiveCommunicator> create_with_same_type(
        communicator base, index_map_ptr imap) const = 0;

    // Construct the inverse pattern (swaps send <-> recv).
    virtual std::unique_ptr<CollectiveCommunicator> create_inverse() const = 0;
};
```

`i_all_to_all_v` returns an `mpi::request`; the caller `wait()`s on it before reading the receive
buffer. Implementations override `i_all_to_all_v_impl` (the typed `void*` form) to dispatch to the
actual MPI call.

`create_inverse()` is what `RowGatherer` uses to derive the *send* index pattern from the *receive*
index pattern stored in the matrix's `IndexMap` — see
[RowGatherer and RowScatterer](row-gather-scatter.md).

## Picking an implementation

The two concrete implementations differ in how MPI sees the communication pattern:

| Implementation | Underlying MPI primitive | Topology view | Best for |
|---|---|---|---|
| `DenseCommunicator` *(default)* | `MPI_Ialltoallv` (with a blocking `MPI_Alltoallv` fallback on Open MPI < 4.1) | Full rank set | Small or moderately-sized rank counts; portable; the pattern Ginkgo defaults to. |
| `NeighborhoodCommunicator` | `MPI_Ineighbor_alltoallv` over a distributed graph topology | Only the actual neighbours | Large rank counts with sparse halo connectivity (typical for FE / FD meshes). |

Both satisfy the same interface, so the choice is transparent to the rest of Ginkgo. The default
selected by `mpi::detail::create_default_collective_communicator(comm)` is the
`DenseCommunicator`. If you want the neighbourhood variant, construct it explicitly and pass it to
`RowGatherer::create(exec, coll_comm, imap)` — the matrix-level wiring picks it up from there.

:::{tip}
The transport choice usually does not change correctness, only performance. Switch to
`NeighborhoodCommunicator` when scaling up: at high rank counts the dense per-rank size vector
becomes the bottleneck, and `MPI_Ineighbor_alltoallv` lets the MPI implementation skip ranks this
process never talks to.
:::

## Implementations

```{toctree}
:maxdepth: 1
:caption: Collective communicators

dense-communicator
neighborhood-communicators
```

:::{seealso}
- [The MPI layer](mpi-layer.md) — `mpi::communicator`, `mpi::request`, the broader collective
  surface this object sits on top of.
- [Communication primitives](communication-primitives.md) — how `i_all_to_all_v` is consumed by
  distributed SpMV.
- [RowGatherer and RowScatterer](row-gather-scatter.md) — the helper that uses a collective
  communicator to perform the halo exchange.
- API reference: [`gko::experimental::mpi::CollectiveCommunicator`](../../../api-reference/mpi/collective-communicator.md)
:::
