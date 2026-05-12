# Communication primitives

A distributed Matrix×Vector apply is the unit of communication in a distributed Ginkgo solver.
This page explains how that operation decomposes into local work and halo exchange, where MPI
happens in the call stack, and the stream-MPI ordering rule for GPU executors.

## Anatomy of a distributed Matrix×Vector

The operation `y = A * x` on distributed objects decomposes into two parts on each rank:

```text
y_local = A_diag     * x_local     ← purely local work
        + A_off_diag * x_halo      ← needs values from other ranks
```

- `A_diag` is the *diagonal matrix*: entries whose row is owned by this rank **and** whose column is also owned by this rank. Sized $n_\text{local rows} \times n_\text{local cols}$, both axes locally indexed.
- `A_off_diag` is the *off-diagonal matrix*: entries whose row is owned by this rank but whose column is owned by some other rank. Stored compressed — only the remote columns this rank's local rows actually reference are kept, and those columns are renumbered $0..K-1$ via the matrix's `index_map`. Sized $n_\text{local rows} \times K$.
- `x_halo` is the value buffer this rank receives during the halo exchange: one entry per remote column the off-diagonal matrix references, ordered to match the compressed column numbering of `A_off_diag`.

Both blocks are accessible through the public API: `mat->get_local_matrix()` returns the
diagonal matrix and `mat->get_non_local_matrix()` returns the off-diagonal matrix. The
"diagonal" prefix refers to the **block structure** of the partitioned global matrix — every
rank's diagonal piece sits on the block-diagonal of the permuted global matrix — not to a
matrix that is literally diagonal.

The halo pattern (which remote rows this rank needs) is derived from `IndexMap` at construction
time and is fixed for the lifetime of the matrix. It is not recomputed on every `apply` call.

## Halo exchange — gather and scatter

The communication protocol for a single distributed apply proceeds as follows:

```text
1. Gather     — each rank packs the x values that its neighbours need into a send buffer.
2. Exchange   — a non-blocking all-to-all-variable (MPI_Alltoallv via i_all_to_all_v) transfers the packed values.
3. Diagonal   — the diagonal matrix apply runs in parallel with the exchange.
4. Wait       — complete the MPI operation, ensuring x_halo is populated.
5. Off-diag   — apply the off-diagonal matrix using the received x_halo values.
6. Sum        — y_local = diagonal-matrix result + off-diagonal-matrix result.
```

The exchange is a single non-blocking variable-sized all-to-all
(`mpi::request i_all_to_all_v(...)` from the collective communicator), not a sequence of
point-to-point `MPI_Isend` / `MPI_Irecv` pairs. This lets MPI implementations pick the optimal
underlying transport — including direct neighbour-to-neighbour sends when the communicator topology
makes that natural — without Ginkgo having to enumerate the send/receive list manually.

In Ginkgo, the gather step is implemented by a `RowGatherer`. The scatter (step 5) uses the
per-`Vector` halo buffer. Both objects are set up once during matrix construction and reused across
all subsequent `apply` calls, so the bookkeeping cost is amortised.

## Where MPI happens

MPI calls are confined to the core distributed layer (`core/distributed/...`). Backend-specific
kernel files (`cuda/`, `hip/`, `dpcpp/`) never include `mpi.h`. This separation is a deliberate
maintainability constraint:

1. The distributed `Matrix::apply` starts the diagonal-matrix kernel on the executor.
2. The `RowGatherer` packs send values into a contiguous buffer (host buffer for standard MPI;
   a device-accessible buffer for CUDA-aware MPI).
3. The collective communicator issues an `i_all_to_all_v` to exchange the packed values.
4. Once the receive completes, the off-diagonal-matrix kernel fires on the executor.

GPU kernels never depend on MPI, and host-side MPI code never touches CUDA or HIP headers. The
consequence is that adding a new GPU backend does not require touching the distributed communication
layer at all.

## Stream-MPI ordering for GPU executors

When using a GPU executor with a non-CUDA-aware MPI, the gather kernel that populates the send
buffer must complete before the host-side MPI call. Without an explicit synchronisation point,
the kernel may still be executing on the device stream when MPI reads the host copy:

```cpp
auto event = exec->create_event();
event->record(exec);
event->synchronize();
// Now safe to launch the host-side i_all_to_all_v on the gathered buffer.
```

A full device sync (`exec->synchronize()`) is the heavier alternative that achieves the same
correctness guarantee.

With a CUDA-aware MPI, the device pointer can be passed directly to the all-to-all call without an
explicit synchronise — the CUDA stream semantics still apply, but MPI manages the ordering.

:::{tip}
Most user code does not orchestrate halo exchange manually. `Matrix::apply` handles all of the
above internally. The ordering details matter for users implementing custom distributed `LinOp`
operations that perform their own MPI communication.
:::

## A diagram of distributed apply

```text
Rank 0                              Rank 1
──────────────────────────          ──────────────────────────
A_diag * x_local (kernel)            A_diag * x_local (kernel)
         |                                   |
         |  halo exchange (MPI_Alltoallv)    |
         |  send boundary x values ────────> |
         | <──────────────────────  send boundary x values
         |                                   |
A_off_diag * x_halo (kernel)         A_off_diag * x_halo (kernel)
         |                                   |
         v                                   v
      y_local                             y_local
```

The diagonal-matrix and off-diagonal-matrix kernels can overlap with the MPI exchange in
time — the diagonal-matrix result does not depend on halo data, so both sides start immediately
while the exchange is in flight.

:::{seealso}
- [The MPI layer](mpi-layer.md) — the communicator, request handle, and collective surface backing
  the call sites above.
- [Collective communicators](collective-communicator.md) — the interface that backs
  `i_all_to_all_v`, plus the dense and neighborhood implementations.
- [RowGatherer and RowScatterer](row-gather-scatter.md) — the helper that packs the send buffer
  and issues the collective.
- [Partitions and index maps](partition-and-index-map.md) — the indexing that drives the halo
  construction.
- [Distributed solvers and preconditioners](solvers-and-preconditioners.md) — what consumes the
  distributed apply.
- API reference: [`gko::experimental::distributed::Matrix`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1experimental_1_1distributed_1_1Matrix.html)
:::
