# Neighborhood communicator

`gko::experimental::mpi::NeighborhoodCommunicator` is a
[CollectiveCommunicator](collective-communicator.md) implementation that performs the halo
exchange with `MPI_Ineighbor_alltoallv` over a distributed-graph topology. Where the
[`DenseCommunicator`](dense-communicator.md) issues a full `P`-way all-to-all and relies on the
MPI runtime to skip zero-size entries, the neighborhood communicator tells MPI up front which
ranks this process actually communicates with, so the all-to-all only involves those neighbours.

## When to use it

- **High rank count with sparse halo connectivity.** This is the typical pattern for
  finite-element and finite-difference simulations: each rank's non-local matrix only
  references values on a handful of neighbouring ranks, regardless of the total job size.
- **The dense per-rank size vector is the bottleneck.** At thousands of ranks, just transmitting
  the full send-size vector to MPI starts to matter. The neighborhood communicator avoids that
  overhead.

For small jobs (≲ a few thousand ranks) the [`DenseCommunicator`](dense-communicator.md) is
simpler, equally correct, and may even be faster because it does not require the MPI runtime to
build a graph topology object.

## Construction

The neighborhood communicator is constructed from a base communicator and an `index_map`:

```cpp
namespace mpi = gko::experimental::mpi;

auto base_comm = mpi::communicator(MPI_COMM_WORLD);

auto coll_comm = std::make_shared<mpi::NeighborhoodCommunicator>(base_comm, imap);
```

Construction does three pieces of work:

1. **Recv-side bookkeeping.** Reads `imap.get_remote_target_ids()` and the segment offsets in
   `imap.get_remote_global_idxs()` to fill in the per-neighbour `recv_sizes_` and the recv target
   id list.
2. **Send-side discovery.** Performs a global `MPI_Alltoall` over a size vector — exactly once,
   at construction — to discover which ranks need to receive *from* this rank, and how many values
   each one wants. This is the unavoidable global step: every rank must learn its inverse
   communication envelope.
3. **Graph topology creation.** Calls `MPI_Dist_graph_create_adjacent` with the recv-from list as
   sources and the send-to list as destinations, producing the topology-aware `MPI_Comm` that
   subsequent `MPI_Ineighbor_alltoallv` calls run over.

Once construction returns, every subsequent `i_all_to_all_v(...)` call only involves the actual
neighbours — typically a small constant per rank, regardless of the total rank count.

A null-pattern constructor `NeighborhoodCommunicator(base_comm)` is also available; it builds an
empty topology so the object is in a valid state before its pattern is set.

## The exchange

```cpp
auto req = coll_comm->i_all_to_all_v(exec, send_ptr, recv_ptr);
// ... overlap work that does not touch recv_ptr ...
req.wait();
```

The primitive is `MPI_Ineighbor_alltoallv`. On Open MPI builds older than 4.1 this primitive is
not safely available with all type combinations and the implementation reports `GKO_NOT_IMPLEMENTED`
— if you hit that, fall back to `DenseCommunicator` or upgrade your MPI.

## Cost model

For $P$ ranks where each rank has on average $k$ neighbours:

- Construction: one global `MPI_Alltoall` of size $P$ integers (the inverse-envelope discovery)
  plus the cost of `MPI_Dist_graph_create_adjacent`. The `Alltoall` is the dominant term and is
  paid **once** per matrix construction.
- Each subsequent exchange: $\Theta(k)$ work outside MPI (scanning the per-neighbour size vector)
  plus $\Theta(N_\text{send})$ for the actual transfer, where $N_\text{send}$ is the total number
  of values this rank sends. Many MPI implementations optimise `MPI_Ineighbor_alltoallv` into
  direct point-to-point sends between neighbours.

So the neighborhood communicator pays a higher **setup** cost (the topology graph creation) in
exchange for a lower **per-exchange** cost. For matrices that are reused across many `apply` calls
— which is the standard case in iterative solvers — the cross-over comes quickly.

## Inverse patterns

`create_inverse()` returns a new `NeighborhoodCommunicator` that swaps the send-from and recv-to
roles. This is what `RowGatherer` uses internally to derive the *send* index pattern from the
*receive* indices stored in the matrix's `index_map`.

:::{seealso}
- [Collective communicators](collective-communicator.md) — the abstract interface and picker
  guidance.
- [Dense communicator](dense-communicator.md) — the portable default.
- [RowGatherer](row-gather-scatter.md) — the helper that uses the inverse
  pattern derived above.
- API reference: [`gko::experimental::mpi::NeighborhoodCommunicator`](../../../api-reference/mpi/neighborhood-communicator.md)
:::
