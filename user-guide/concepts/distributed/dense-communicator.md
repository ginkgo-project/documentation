# Dense communicator

`gko::experimental::mpi::DenseCommunicator` is the default
[CollectiveCommunicator](collective-communicator.md) implementation. It performs the halo
exchange with a blocking `MPI_Alltoallv` over the full rank set: every rank passes the entire per-rank send
size and offset vector to MPI, and ranks that exchange zero bytes simply have a zero entry. The
returned `mpi::request` is already complete. This
is the safe, portable default: it works on every MPI 3.x and 4.x implementation and does not
require the platform's MPI to expose distributed-graph topology support.

## When to use it

- **You are not running at extreme rank counts.** Up to a few thousand ranks, the constant-factor
  overhead of `MPI_Alltoallv` over the full rank set is negligible compared to the actual data
  transfer.
- **You want maximum portability.** Some MPI builds disable or under-implement
  `MPI_Ineighbor_alltoallv` (the alternative); the dense path is always available.
- **You do not yet know your halo connectivity is sparse.** Until profiling shows the all-to-all
  is dominating, the dense pattern is a perfectly reasonable choice.

For high rank counts with sparse halo connectivity (typical FE / FD meshes), prefer
[`NeighborhoodCommunicator`](neighborhood-communicators.md); it lets MPI skip the zero-size
entries entirely.

## Construction

The expected pattern is to derive the communicator from an `index_map`, which already encodes which
remote ranks own which indices this rank needs:

```cpp
namespace mpi = gko::experimental::mpi;

auto base_comm = mpi::communicator(MPI_COMM_WORLD);

auto coll_comm = std::make_shared<mpi::DenseCommunicator>(base_comm, imap);
```

The constructor reads the index map's `remote_target_ids` and `remote_global_idxs` to compute the
recv-side per-rank sizes, then performs an `MPI_Alltoall` over the size vector to discover the
send-side sizes. After this point the communicator is "primed": every subsequent
`i_all_to_all_v(...)` call uses the cached size and offset vectors.

A null-pattern constructor is also available: `DenseCommunicator(base_comm)` builds a
communicator with empty send and receive sets. It exists mainly so that distributed types can be
default-constructed before their pattern is known.

## What it stores

Internally the dense communicator caches four `std::vector<comm_index_type>` per rank:

- `send_sizes_` and `send_offsets_`: how many values this rank sends to each peer rank, and the
  start index of that block in the send buffer.
- `recv_sizes_` and `recv_offsets_`: the receive-side equivalents.

`get_send_size()` and `get_recv_size()` return the cumulative totals (the back element of the
respective offsets vector), which is the exact buffer size each side needs to allocate. The
matrix's `RowGatherer` reads these to size the contiguous send / receive buffers it manages.

## The exchange

```cpp
auto req = coll_comm->i_all_to_all_v(exec, send_ptr, recv_ptr);
req.wait();
```

`i_all_to_all_v` calls the blocking `MPI_Alltoallv` with the cached size and offset vectors and
returns an `mpi::request` that is already complete, so the exchange finishes before the call
returns and work issued after the call does not overlap with it (on device executors the
distributed `Matrix::apply` queues the local kernel before issuing the exchange, so that kernel
does overlap). Calling `wait()` on the returned request is still the right pattern.

## Cost model

For $P$ ranks, the per-call work is:

- $\Theta(P)$ to scan the size and offset vectors on each rank (small, typically cache-resident).
- $\Theta(N_\text{send})$ to actually transfer the values, where $N_\text{send}$ is the total
  number of values this rank sends.

The `MPI_Alltoallv` collective itself is internally implemented by most MPI runtimes as either a
direct exchange between active senders and receivers (fast for sparse patterns) or a reduction
tree (better for dense patterns). The Ginkgo side does not control the choice; it just supplies
the per-rank size vector and lets MPI pick.

:::{seealso}
- [Collective communicators](collective-communicator.md): the abstract interface, picker
  guidance, and the inverse-pattern protocol.
- [Neighborhood communicator](neighborhood-communicators.md): the sparse-topology alternative.
- [Communication primitives](communication-primitives.md): how the dense `i_all_to_all_v` is
  consumed by distributed SpMV.
- API reference: [`gko::experimental::mpi::DenseCommunicator`](../../../api-reference/mpi/dense-communicator.md)
:::
