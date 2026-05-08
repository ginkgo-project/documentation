# RowGatherer

`gko::experimental::distributed::RowGatherer<L>` is the helper that drives the halo exchange in
distributed `Matrix::apply`. It packs the rows of a distributed `Vector` that other ranks need
into a contiguous buffer, fires the all-to-all through a
[CollectiveCommunicator](collective-communicator.md), and writes the result into a destination
vector. Most user code does not interact with it directly — the distributed `Matrix` constructs
and uses one internally — but it is also exposed publicly for users who want to perform their own
asynchronous halo exchange in custom distributed `LinOp` operations.

## What it does

The mathematical operation is the row-selection map
$$
x_{\text{halo}} = R_{\text{halo}} \, x
$$
where `R_halo` is the indicator that picks out the global rows of `x` that this rank's ghost
matrix needs. In distributed terms each rank owns a contiguous block of `x`, so satisfying
$x_{\text{halo}}$ requires **other** ranks to send their owned values to this rank — i.e. an
all-to-all. `RowGatherer` is the object that owns the index list ("which local rows of *my* `x`
do I need to send out, and to whom") plus the send-side buffer.

## Construction

The recommended constructor takes an executor, a collective communicator, and an `IndexMap`:

```cpp
namespace dist = gko::experimental::distributed;
namespace mpi  = gko::experimental::mpi;

// Build a collective communicator from the matrix's index map.
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
auto b = dist::Vector<double>::create(exec, comm, ...);
auto x = gko::matrix::Dense<double>::create(exec, ...);

auto req = rg->apply_async(b, x);
// ... do unrelated work that does not touch x ...
req.wait();
// x now holds the gathered rows.
```

`apply_async` returns an `mpi::request`. The receive buffer (`x`) must not be read or written
between the call and `req.wait()`, and the sender's local data inside `b` must not be modified
either. Once `wait()` returns, `x` holds exactly `get_size()[0]` rows — the rows of `b` that the
collective communicator's pattern said this rank needs.

:::{warning}
Only one `mpi::request` per `RowGatherer` instance can be active at a time. Calling `apply_async`
again before `wait()`-ing on the previous request is undefined behaviour because the internal
send buffer is reused. If you need overlapping exchanges, build multiple gatherers (one per
in-flight operation), or pass a separate `GenericDenseCache` workspace through the three-argument
overload.
:::

The output executor must be compatible with the MPI implementation. If the MPI is not
GPU-aware and the executor is a `CudaExecutor` / `HipExecutor` / `DpcppExecutor`, the call throws
unless the destination's executor is the host executor. Internally, `RowGatherer` uses
`mpi::requires_host_buffer(exec, comm)` to decide whether to stage through host memory.

## Inspection accessors

`RowGatherer` exposes a small read-only surface for inspecting the cached pattern:

| Method | Returns |
|---|---|
| `get_size()` | `dim<2>` — total recv rows (rows) × global vector width (cols) |
| `get_collective_communicator()` | The shared `CollectiveCommunicator` driving the exchange |
| `get_const_send_idxs()` | Pointer to the local row indices being sent |
| `get_num_send_idxs()` | Length of the send-index array |

The send-index array is stored on the same executor the row gatherer lives on, so reading it from
the host requires `gko::clone(host, ...)` first.

## How distributed `Matrix::apply` uses it

The matrix holds a `RowGatherer` per `IndexMap`. The 6-step apply trace
([Communication primitives](communication-primitives.md)) calls into it as follows:

1. `rg->apply_prepare(b)` packs the values that need to be sent into the workspace buffer.
2. The local matrix kernel runs in parallel.
3. `rg->apply_finalize(b, x_halo, prepare_event)` issues the all-to-all and returns the
   `mpi::request`.
4. `req.wait()` — `x_halo` is now populated.
5. The ghost matrix kernel applies to `x_halo`.

These two protected entry points (`apply_prepare` / `apply_finalize`) are exposed via the
`detail` friend interface for testing and for advanced users who want fine-grained control over
when the exchange is launched relative to other work.

## RowScatterer

A scatter-side counterpart — `RowScatterer` — is the natural complement of `RowGatherer`: it
takes a contiguous buffer of values addressed in the local-to-global send pattern and accumulates
them into the right rows of a distributed `Vector` on the receiver side. This is the operation
needed for adjoint apply, sparse matrix–transpose products, and certain reduction-style
preconditioners.

:::{note}
`RowScatterer` is not part of the current public API. The all-to-all primitive in
[`CollectiveCommunicator`](collective-communicator.md) is symmetric (the inverse pattern obtained
through `create_inverse()` already realises the scatter direction at the MPI level), so the
operation is *expressible* today by building an inverse `RowGatherer`. A dedicated `RowScatterer`
type is a planned addition that will encapsulate the inverse-direction bookkeeping the same way
`RowGatherer` does for the gather direction.
:::

:::{seealso}
- [Communication primitives](communication-primitives.md) — the SpMV pipeline that consumes the
  row gatherer.
- [Collective communicators](collective-communicator.md) — the layer beneath the gatherer.
- [Partitions and index maps](partition-and-index-map.md) — the bookkeeping that defines which
  rows get gathered.
- API reference: [`gko::experimental::distributed::RowGatherer`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1experimental_1_1distributed_1_1RowGatherer.html)
:::
