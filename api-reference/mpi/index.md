# `gko::experimental::mpi`

Thin C++ wrappers around the MPI runtime. The
`gko::experimental::distributed` types are built on top of these
primitives: a `Matrix` or `Vector` carries an `mpi::communicator`,
nonblocking operations return an `mpi::request`, and the
collective-communicator hierarchy abstracts the actual data exchange
patterns used by the distributed SpMV.

:::{attention}
Everything in `gko::experimental::mpi` is experimental: the interfaces
may change between releases.
:::

Most application code only ever touches `environment` (RAII wrapper
around `MPI_Init` / `MPI_Finalize`) and `communicator` (wrapper
around `MPI_Comm`). The remaining types are the lower-level
building blocks that the distributed kernels use internally.

## Runtime

- [`environment`](environment.md): RAII wrapper around `MPI_Init` and
  `MPI_Finalize`; construct once at the top of `main`.
- [`communicator`](communicator.md): reference-counted C++ wrapper
  around `MPI_Comm` that the distributed `Matrix`, `Vector` and `RowGatherer` hold onto.

## Asynchronous operations

- [`request`](request.md): move-only handle representing an
  outstanding nonblocking MPI operation; analogous to `MPI_Request`.
- [`status`](status.md): light wrapper around `MPI_Status`, returned
  by completion calls.

## Type system

- [`contiguous_type`](contiguous-type.md): RAII wrapper that builds
  and frees a contiguous `MPI_Datatype` for sending fixed-size blocks
  of an element type.

## Collective communicators

The distributed SpMV's halo exchange is abstracted as a
`CollectiveCommunicator`. Ginkgo ships two concrete implementations;
applications can plug in their own.

- [`CollectiveCommunicator`](collective-communicator.md): abstract
  base for gather/scatter on a `RowGatherer`-style index pattern.
- [`DenseCommunicator`](dense-communicator.md): implementation using
  `MPI_Alltoallv`-style dense collectives. Default everywhere; works
  on any communicator topology.
- [`NeighborhoodCommunicator`](neighborhood-communicator.md):
  implementation using MPI-3 neighborhood collectives over a
  graph-topology communicator. Faster on large rank counts where each
  rank only talks to a small set of neighbours.

```{toctree}
:hidden:
:maxdepth: 1

environment
communicator
request
status
contiguous-type
collective-communicator
dense-communicator
neighborhood-communicator
```
