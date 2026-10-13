# `gko::experimental::mpi::`<br>`CollectiveCommunicator`

Abstract interface for the collective communication patterns used by
the distributed SpMV. A `CollectiveCommunicator` wraps a base
`communicator`; the concrete implementations also take an `index_map`
that defines the communication pattern. It exposes a single
nonblocking variable all-to-all primitive that ships locally-owned
values to the ranks that need them as ghost entries.

The distributed `Matrix` holds a `CollectiveCommunicator` internally
and invokes `i_all_to_all_v` to gather the halo values that the off-diagonal SpMV consumes; the
exchange overlaps the diagonal SpMV (on host executors only when the exchange is non-blocking; see
the Open MPI < 4.1 fallback of [`DenseCommunicator`](dense-communicator.md)) and completes before
the off-diagonal SpMV starts. Two
concrete implementations ship with Ginkgo —
[`DenseCommunicator`](dense-communicator.md) and
[`NeighborhoodCommunicator`](neighborhood-communicator.md) — and
applications can supply their own by deriving from this interface.

```{doxygenclass} gko::experimental::mpi::CollectiveCommunicator
:members:
```
