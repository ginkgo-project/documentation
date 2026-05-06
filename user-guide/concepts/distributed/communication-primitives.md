# Communication primitives

A distributed Matrix×Vector apply is the unit of communication in a distributed Ginkgo solver.
This page explains how that operation decomposes into local work and halo exchange, where MPI
happens in the call stack, and the stream-MPI ordering rule for GPU executors.

## Anatomy of a distributed Matrix×Vector

The operation `y = A * x` on distributed objects decomposes into two parts on each rank:

```text
y_local = A_diag    * x_local        ← purely local work
        + A_offdiag * x_halo         ← needs values from other ranks
```

- `A_diag` is the on-diagonal block: rows and columns both owned by this rank.
- `A_offdiag` is the off-diagonal block: rows owned by this rank, but columns owned by other ranks.
- `x_halo` is the portion of `x` owned by other ranks that this rank's off-diagonal block references.

The halo pattern (which remote rows this rank needs) is derived from `IndexMap` at construction
time and is fixed for the lifetime of the matrix. It is not recomputed on every `apply` call.

## Halo exchange — gather and scatter

The communication protocol for a single distributed apply proceeds as follows:

```text
1. Gather   — each rank packs the x values that its neighbours need into a send buffer.
2. Exchange — non-blocking MPI_Isend / MPI_Irecv transfer the packed values.
3. Local    — the on-diagonal block apply runs in parallel with the exchange.
4. Wait     — complete the MPI operations, ensuring x_halo is populated.
5. Offdiag  — apply the off-diagonal block using the received x_halo values.
6. Sum      — y_local = diag result + offdiag result.
```

In Ginkgo, the gather step is implemented by a `RowGatherer`. The scatter (step 5) uses the
per-`Vector` halo buffer. Both objects are set up once during matrix construction and reused across
all subsequent `apply` calls, so the bookkeeping cost is amortised.

## Where MPI happens

MPI calls are confined to the core distributed layer (`core/distributed/...`). Backend-specific
kernel files (`cuda/`, `hip/`, `dpcpp/`) never include `mpi.h`. This separation is a deliberate
maintainability constraint:

1. The distributed `Matrix::apply` starts the local on-diagonal kernel on the executor.
2. The `RowGatherer::gather` packs send values into a contiguous buffer (host buffer for standard
   MPI; a device-accessible buffer for CUDA-aware MPI).
3. `MPI_Isend` and `MPI_Irecv` transfer the data.
4. Once the receive completes, the off-diagonal kernel fires on the executor.

GPU kernels never depend on MPI, and host-side MPI code never touches CUDA or HIP headers. The
consequence is that adding a new GPU backend does not require touching the distributed communication
layer at all.

## Stream-MPI ordering for GPU executors

When using a GPU executor with a non-CUDA-aware MPI, the gather kernel that populates the send
buffer must complete before the host-side `MPI_Isend`. Without an explicit synchronisation point,
the kernel may still be executing on the device stream when MPI reads the host copy:

```cpp
auto event = exec->create_event();
event->record(exec);
event->synchronize();
// Now safe to call MPI_Isend on the gathered buffer.
```

A full device sync (`exec->synchronize()`) is the heavier alternative that achieves the same
correctness guarantee.

With a CUDA-aware MPI, the device pointer can be passed directly to `MPI_Isend` without an
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
A_diag * x_local (kernel)           A_diag * x_local (kernel)
         |                                   |
         |     halo exchange (MPI)           |
         |  send boundary x values ────────> |
         | <──────────────────────  send boundary x values
         |                                   |
A_offdiag * x_halo (kernel)         A_offdiag * x_halo (kernel)
         |                                   |
         v                                   v
      y_local                             y_local
```

The on-diagonal and off-diagonal kernels can overlap with the MPI exchange in time — the
on-diagonal result does not depend on halo data, so both sides start immediately while the
exchange is in flight.

:::{seealso}
- [Partitions and index maps](partition-and-index-map.md) — the indexing that drives the halo
  construction.
- [Distributed solvers and preconditioners](solvers-and-preconditioners.md) — what consumes the
  distributed apply.
- API reference: [`gko::experimental::distributed::Matrix`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1experimental_1_1distributed_1_1Matrix.html)
:::
