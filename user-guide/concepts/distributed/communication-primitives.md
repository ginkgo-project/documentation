# Communication primitives

A distributed Matrix×Vector apply is the unit of communication in a distributed Ginkgo solver.
This page explains how that operation decomposes into local work and halo exchange, where MPI
happens in the call stack, and the stream-MPI ordering rule for GPU executors.

## Anatomy of a distributed Matrix×Vector

The operation `y = A * x` on distributed objects decomposes into two parts on each rank:

```text
y_local = A_local     * x_local    ← purely local work
        + A_non_local * x_halo     ← needs values from other ranks
```

- `A_local` is the *local matrix*: entries whose row is owned by this rank **and** whose column is also owned by this rank. Sized $n_\text{local rows} \times n_\text{local cols}$, both axes locally indexed.
- `A_non_local` is the *non-local matrix*: entries whose row is owned by this rank but whose column is owned by some other rank. Stored compressed — only the remote columns this rank's local rows actually reference are kept, and those columns are renumbered $0..K-1$ via the matrix's `index_map`. Sized $n_\text{local rows} \times K$.
- `x_halo` is the value buffer this rank receives during the halo exchange: one entry per remote column the non-local matrix references, ordered to match the compressed column numbering of `A_non_local`.

Both blocks are accessible through the public API: `mat->get_local_matrix()` returns the
local (diagonal-block) matrix and `mat->get_non_local_matrix()` returns the non-local
(off-diagonal-block) matrix. The "diagonal" and "off-diagonal" wording refers to the **block
structure** of the partitioned global matrix — every rank's local piece sits on the
block-diagonal of the permuted global matrix — not to a matrix that is literally diagonal.

The halo pattern (which remote rows this rank needs) is derived from the `index_map` at construction
time and is fixed for the lifetime of the matrix. It is not recomputed on every `apply` call.

## Halo exchange — gather and scatter

The communication protocol for a single distributed apply proceeds as follows:

```text
1. Gather     — each rank packs the x values that its neighbours need into a send buffer.
2. Exchange   — a non-blocking all-to-all-variable (MPI_Alltoallv via i_all_to_all_v) transfers the packed values.
3. Local      — the local matrix apply runs in parallel with the exchange.
4. Wait       — complete the MPI operation, ensuring x_halo is populated.
5. Non-local  — apply the non-local matrix using the received x_halo values.
6. Sum        — y_local = local-matrix result + non-local-matrix result.
```

The exchange is a single non-blocking variable-sized all-to-all
(`mpi::request i_all_to_all_v(...)` from the collective communicator), not a sequence of
point-to-point `MPI_Isend` / `MPI_Irecv` pairs. This lets MPI implementations pick the optimal
underlying transport — including direct neighbour-to-neighbour sends when the communicator topology
makes that natural — without Ginkgo having to enumerate the send/receive list manually.

In Ginkgo, the gather step is implemented by a `RowGatherer`. The receive side (step 5) uses a
halo buffer cached inside the `Matrix` and resized on demand for the number of right-hand-side
columns. The `RowGatherer` is set up by `read_distributed` and reused across all subsequent
`apply` calls, so the bookkeeping cost is amortised.

## Where MPI happens

MPI calls live in host-side library code: the distributed classes under `core/distributed`, the
`Gmres` reduction and the aggregate exchange of `Pgm`. Backend-specific kernel files (`cuda/`,
`hip/`, `dpcpp/`) never include `mpi.h`. This separation is a deliberate maintainability
constraint. Within a distributed `Matrix::apply`:

1. The `RowGatherer` packs send values into a contiguous buffer (host buffer for standard MPI;
   a device-accessible buffer for CUDA-aware MPI).
2. The collective communicator issues an `i_all_to_all_v` to exchange the packed values; on
   device executors the local-matrix kernel is submitted before the exchange is issued, so it
   does not wait for the gather.
3. The local-matrix kernel runs while the exchange is in flight.
4. Once the receive completes, the non-local-matrix kernel fires on the executor.

GPU kernels never depend on MPI, and host-side MPI code never touches CUDA or HIP headers. The
consequence is that adding a new GPU backend does not require touching the distributed communication
layer at all.

## Stream-MPI ordering for GPU executors

When using a GPU executor with a non-GPU-aware MPI, the gather kernel that populates the send
buffer runs on the executor's device stream while the MPI call runs on the host. The kernel
must finish before the host-side MPI call reads the staged buffer. Without an explicit
synchronisation point, the kernel may still be executing on the device stream when MPI starts
reading.

The straightforward way to enforce the ordering in user code is a full executor sync after the
kernel and before the MPI call:

```cpp
exec->synchronize();
// Now safe to launch the host-side i_all_to_all_v on the gathered buffer.
```

This waits for everything queued on the executor's stream, not just the gather kernel. Internally
Ginkgo's `RowGatherer` does something more granular — it records an event on the stream right
after the gather kernel and waits on just that event before issuing the all-to-all — but the
event-recording machinery (`gko::detail::Event`, the `event::record_event` kernel registered in
`core/distributed/row_gatherer.cpp`) is internal-by-convention and not a stable user-facing API.
For an external distributed `LinOp`, the executor-level sync is the supported path.

With a GPU-aware MPI build (`-DGINKGO_FORCE_GPU_AWARE_MPI=ON` at configure time), the gathered
device buffer is handed to MPI directly, without staging through host memory. The `RowGatherer`
still waits on the gather event before it issues the all-to-all, so the send buffer is complete
when MPI reads it.

:::{tip}
Most user code does not orchestrate halo exchange manually. `Matrix::apply` handles all of the
above internally. The ordering details matter for users implementing custom distributed `LinOp`
operations that perform their own MPI communication.
:::

## A diagram of distributed apply

```text
Rank 0                              Rank 1
──────────────────────────          ──────────────────────────
A_local * x_local (kernel)           A_local * x_local (kernel)
         |                                   |
         |  halo exchange (MPI_Alltoallv)    |
         |  send boundary x values ────────> |
         | <──────────────────────  send boundary x values
         |                                   |
A_non_local * x_halo (kernel)        A_non_local * x_halo (kernel)
         |                                   |
         v                                   v
      y_local                             y_local
```

The local-matrix kernel can overlap with the MPI exchange in time — its result does not depend
on halo data, so both sides start immediately while the exchange is in flight.

:::{seealso}
- [The MPI layer](mpi-layer.md) — the communicator, request handle, and collective surface backing
  the call sites above.
- [Collective communicators](collective-communicator.md) — the interface that backs
  `i_all_to_all_v`, plus the dense and neighborhood implementations.
- [RowGatherer](row-gather-scatter.md) — the helper that packs the send buffer
  and issues the collective.
- [Partitions and index maps](partition-and-index-map.md) — the indexing that drives the halo
  construction.
- [Distributed solvers and preconditioners](solvers-and-preconditioners.md) — what consumes the
  distributed apply.
- API reference: [`gko::experimental::distributed::Matrix`](../../../api-reference/distributed/matrix.md)
:::
