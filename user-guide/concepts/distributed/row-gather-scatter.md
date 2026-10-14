# RowGatherer

Distributed sparse-matrix kernels need two symmetric pieces of MPI bookkeeping. To compute
$y = A x$ where $A$ is row-distributed, every rank must **pull in** the entries of $x$ that its
off-diagonal matrix references but does not own (a halo gather). To compute $y = A^T x$ (or any
operation that builds $y$ from contributions on multiple ranks), every rank must **push out**
locally-computed contributions to the ranks that own the corresponding rows of $y$, where they are
accumulated (a halo scatter).

`gko::experimental::distributed::RowGatherer<L>` is the helper object that encapsulates the gather
pattern and drives the halo exchange in distributed `Matrix::apply`. It packs the rows of a
distributed `Vector` that other ranks need into a contiguous buffer, fires the all-to-all through a
[collective communicator](collective-communicator.md), and writes the result into a destination
vector. It reuses a precomputed index pattern across many `apply` calls so that the per-call cost is
just the all-to-all itself. The scatter pattern is
the gather pattern with send and receive roles swapped, which is exactly what
`CollectiveCommunicator::create_inverse()` produces.

Most user code does not interact with these helpers directly; the distributed `Matrix` constructs
and uses them internally. They are exposed publicly for users implementing custom distributed
`LinOp` operations that need to drive their own asynchronous halo exchanges.

## What it does

The mathematical operation is the row-selection map $x_\text{halo} = R_\text{halo}\, x$, where
$R_\text{halo}$ is the indicator that picks out the global rows of $x$ that this rank's
off-diagonal matrix needs. In distributed terms each rank owns a contiguous block of $x$, so satisfying
$x_\text{halo}$ requires **other** ranks to send their owned values to this rank, i.e. an
all-to-all. `RowGatherer` is the object that owns the index list ("which local rows of *my* $x$
do I need to send out, and to whom") plus the send-side buffer.

## Construction

The recommended constructor takes an executor, a collective communicator, and an `index_map`:

```cpp
namespace dist = gko::experimental::distributed;
namespace mpi  = gko::experimental::mpi;

// imap: an index_map built from the partition, this rank's id and the
// global indices this rank accesses but does not own.
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

`RowGatherer::create(exec, coll_comm_template)` builds an empty gatherer whose communicator has the
runtime type of `coll_comm_template`. Passed to `Matrix::create(exec, rg_template)`, it selects the
communicator type that `read_distributed` uses (see
[Collective communicators](collective-communicator.md)). `Matrix::get_row_gatherer()` returns the
gatherer a matrix uses, so `A->get_row_gatherer()->get_collective_communicator()` shows which
exchange it performs.

## The asynchronous apply

```cpp
// b: a dist::Vector<double> holding the values to gather.
auto x = dist::Vector<double>::create(
    exec, comm,
    gko::dim<2>{rg->get_size()[0], 1},      // global: all ranks' received rows
    gko::dim<2>{static_cast<gko::size_type>(
        rg->get_collective_communicator()->get_recv_size()), 1});  // local: rows this rank receives

auto req = rg->apply_async(b, x);
// ... do unrelated work that does not touch x ...
req.wait();
// The local part of x now holds the gathered rows.
```

`apply_async` returns an `mpi::request`. The receive buffer (`x`) must not be read or written
between the call and `req.wait()`, and the sender's local data inside `b` must not be modified
either. Once `wait()` returns, the local part of `x` holds the `get_recv_size()` rows of `b` that this
rank's pattern requests. The output `x` is a distributed `Vector`: the gatherer writes into its
local values.

:::{attention}
The two-argument `apply_async` uses the gatherer's internal send buffer, so only one `mpi::request`
per `RowGatherer` instance can be active at a time. Calling it again before `wait()`-ing on the
previous request is undefined behaviour. To overlap multiple exchanges on the same gatherer, use
the three-argument overload and pass a separate `GenericDenseCache` workspace per in-flight call.
:::

The output executor must be compatible with the MPI implementation. If the MPI is not
GPU-aware and the executor is a `CudaExecutor` / `HipExecutor` / `DpcppExecutor`, the call throws
unless the destination's executor is the host executor. Internally, `RowGatherer` uses
`mpi::requires_host_buffer(exec, comm)` to decide whether to stage through host memory.

## Inspection accessors

`RowGatherer` exposes a small read-only surface for inspecting the cached pattern:

| Method | Returns |
|---|---|
| `get_size()` | `dim<2>` (rows: total received rows summed over all ranks; cols: global size of the gathered index space) |
| `get_collective_communicator()` | The shared `CollectiveCommunicator` driving the exchange |
| `get_const_send_idxs()` | Pointer to the local row indices being sent |
| `get_num_send_idxs()` | Length of the send-index array |

The send-index array is stored on the same executor the row gatherer lives on, so reading it from
the host requires `gko::clone(host, ...)` first.

## How distributed `Matrix::apply` uses it

The matrix holds a `RowGatherer` built from its `index_map`. On device executors, `Matrix::apply`
calls into it in this order (step names from the 6-step trace in
[Communication primitives](communication-primitives.md)):

1. `rg->apply_prepare(b)` packs the values that need to be sent into the workspace buffer and
   returns an event (trace step 1, Gather).
2. The diagonal-matrix kernel is queued on the executor (trace step 3, Diagonal), so its launch does
   not wait for the gather event.
3. `rg->apply_finalize(b, x_halo, prepare_event)` waits on the gather event, issues the all-to-all
   and returns the `mpi::request` (trace step 2, Exchange).
4. `req.wait()`: `x_halo` is now populated (trace step 4).
5. The off-diagonal-matrix kernel applies to `x_halo`.
6. The diagonal-matrix result and the off-diagonal-matrix result are summed into `y_local`.

On the host executors (Reference and OMP) the matrix issues `apply_async` first and then runs the
diagonal-matrix kernel while the exchange is in flight (for a non-blocking communicator; with the
blocking Open MPI < 4.1 fallback of the [Dense communicator](dense-communicator.md) the exchange has
already completed). The two protected entry points (`apply_prepare` / `apply_finalize`) are exposed
via the `detail` friend interface for testing and for advanced users who want fine-grained control
over when the exchange is launched relative to other work.

:::{seealso}
- [Communication primitives](communication-primitives.md): the SpMV pipeline that consumes the
  row gatherer.
- [Collective communicators](collective-communicator.md): the layer beneath the gatherer.
- [Partitions and index maps](partition-and-index-map.md): the bookkeeping that defines which
  rows get gathered.
- API reference: [`gko::experimental::distributed::RowGatherer`](../../../api-reference/distributed/row-gatherer.md)
:::
