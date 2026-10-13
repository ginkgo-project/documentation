# `gko::experimental::mpi::`<br>`CollectiveCommunicator`

Abstract interface for the collective communication patterns used by
the distributed SpMV. A `CollectiveCommunicator` wraps a base
`communicator` (the concrete implementations also take an `index_map`
that defines the pattern) and exposes a single nonblocking variable
all-to-all primitive that ships locally-owned values to the ranks that
need them as ghost entries.

The distributed `Matrix` holds a `CollectiveCommunicator` internally
and invokes `i_all_to_all_v` during the non-local SpMV. Two
concrete implementations ship with Ginkgo —
[`DenseCommunicator`](dense-communicator.md) and
[`NeighborhoodCommunicator`](neighborhood-communicator.md) — and
applications can supply their own by deriving from this interface.

```{doxygenclass} gko::experimental::mpi::CollectiveCommunicator
:members:
```
