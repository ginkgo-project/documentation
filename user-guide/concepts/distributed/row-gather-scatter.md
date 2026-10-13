# RowGatherer

To compute $y = A x$ where $A$ is row-distributed, every rank must **pull in** the entries of
$x$ that its non-local matrix references but does not own — a halo gather. `RowGatherer` is the
helper object that encapsulates this pattern. It sits on top of a
[collective communicator](collective-communicator.md) and reuses a precomputed index pattern
across many `apply` calls so that the per-call cost is just the all-to-all itself. Internally it uses
`CollectiveCommunicator::create_inverse()` to turn each rank's receive pattern into the matching
send pattern.

Most user code does not interact with the gatherer directly — the distributed `Matrix` constructs
and uses it internally. `gko::experimental::distributed::RowGatherer<L>` is public for users
implementing custom distributed `LinOp` operations that drive their own asynchronous halo
exchanges: it packs the rows of a distributed `Vector` that other ranks need into a contiguous
buffer, starts the all-to-all through its collective communicator, and writes the result into a
destination vector.

## What it does

The mathematical operation is the row-selection map $x_\text{halo} = R_\text{halo}\, x$, where
$R_\text{halo}$ is the indicator that picks out the global rows of $x$ that this rank's
non-local matrix needs. In distributed terms each rank owns a block of $x$, so satisfying
$x_\text{halo}$ requires **other** ranks to send their owned values to this rank — i.e. an
all-to-all. `RowGatherer` is the object that owns the index list ("which local rows of *my* $x$
do I need to send out, and to whom") plus the send-side buffer.

## Construction

The recommended constructor takes an executor, a collective communicator, and an `index_map`:

```cpp
namespace dist = gko::experimental::distributed;
namespace mpi  = gko::experimental::mpi;

// imap: the index_map of the receive pattern, built from the partition, this rank's id and the
// global indices it accesses but does not own:
//   dist::index_map<gko::int32, gko::int64> imap{exec, partition, comm.rank(), recv_connections};
// Build a collective communicator from it.
auto coll_comm = std::make_shared<mpi::NeighborhoodCommunicator>(comm, imap);

// Construct the row gatherer.
auto rg = dist::RowGatherer<gko::int32>::create(exec, coll_comm, imap);
```

Internally the constructor calls `coll_comm->create_inverse()` and runs a single
`i_all_to_all_v` to derive the local **send** indices from the remote **recv** indices encoded in
the index map. The send-index array is cached on the executor and reused for every subsequent
exchange.

A simpler `RowGatherer::create(exec, comm)` constructor is available for cases where you only have
a base communicator. It builds an empty default `CollectiveCommunicator` (currently
`DenseCommunicator`) and is mainly useful as a default-constructed handle to be assigned into.

## The asynchronous apply

```cpp
// b: the distributed input; x: the receive vector, one local row per gathered row.
auto x = dist::Vector<double>::create(
    exec, comm, gko::dim<2>{rg->get_size()[0], 1},
    gko::dim<2>{static_cast<gko::size_type>(coll_comm->get_recv_size()), 1});

auto req = rg->apply_async(b, x);
// ... do unrelated work that does not touch x ...
req.wait();
// x now holds the gathered rows.
```

`apply_async` returns an `mpi::request`. The receive vector `x` is a distributed `Vector` (the
implementation writes its local part, even though the class documentation describes it
as a `matrix::Dense`): its global row count is `rg->get_size()[0]` and each rank's local row count
is the communicator's `get_recv_size()`. The receive vector must not be read or written between
the call and `req.wait()`, and the sender's local data inside `b` must not be modified either.
Once `wait()` returns, the local part of `x` holds the rows of `b` that the collective
communicator's pattern said this rank needs.

:::{attention}
The two-argument `apply_async` uses the gatherer's internal send buffer, so only one `mpi::request`
per `RowGatherer` instance can be active at a time. Calling it again before `wait()`-ing on the
previous request is undefined behaviour. With a non-blocking collective communicator, overlap multiple exchanges on the same gatherer by using
the three-argument overload and passing a separate `array<char>` workspace per in-flight call.
:::

The output executor must be compatible with the MPI implementation. If the MPI is not
GPU-aware and the executor is a `CudaExecutor` / `HipExecutor` / `DpcppExecutor`, the call throws
unless the destination's executor is the host executor. Internally, `RowGatherer` uses
`mpi::requires_host_buffer(exec, comm)` to decide whether to stage through host memory.

## Inspection accessors

`RowGatherer` exposes a small read-only surface for inspecting the cached pattern:

| Method | Returns |
|---|---|
| `get_size()` | `dim<2>` — recv rows summed over all ranks (rows) × global index-space size of the index map (cols) |
| `get_collective_communicator()` | The shared `CollectiveCommunicator` driving the exchange |
| `get_const_send_idxs()` | Pointer to the local row indices being sent |
| `get_num_send_idxs()` | Length of the send-index array |

The send-index array is stored on the same executor the row gatherer lives on, so reading it from
the host requires `gko::clone(host, ...)` first.

## How distributed `Matrix::apply` uses it

The matrix holds one `RowGatherer` together with its `index_map`. On device executors the
6-step apply trace ([Communication primitives](communication-primitives.md)) calls into it as
follows; on the host executor the matrix issues `apply_async` first and then runs the local-matrix
kernel (with the default `DenseCommunicator` the exchange is blocking and has already completed by
then; a non-blocking communicator overlaps it with the kernel):

1. `rg->apply_prepare(b)` packs the values that need to be sent into the workspace buffer.
2. The local-matrix kernel runs in parallel.
3. `rg->apply_finalize(b, x_halo, prepare_event)` issues the all-to-all and returns the
   `mpi::request`.
4. `req.wait()` — `x_halo` is now populated.
5. The non-local-matrix kernel applies to `x_halo`.
6. The local-matrix result and the non-local-matrix result are summed into `y_local`.

These two protected entry points (`apply_prepare` / `apply_finalize`) are exposed via the
`detail` friend interface for testing and for advanced users who want fine-grained control over
when the exchange is launched relative to other work.

:::{seealso}
- [Communication primitives](communication-primitives.md) — the SpMV pipeline that consumes the
  row gatherer.
- [Collective communicators](collective-communicator.md) — the layer beneath the gatherer.
- [Partitions and index maps](partition-and-index-map.md) — the bookkeeping that defines which
  rows get gathered.
- API reference: [`gko::experimental::distributed::RowGatherer`](../../../api-reference/distributed/row-gatherer.md)
:::
