# `gko::experimental::mpi::`<br>`DenseCommunicator`

Default implementation of
[`CollectiveCommunicator`](collective-communicator.md). Uses
a blocking `MPI_Alltoallv` over the standard MPI communicator (no
graph topology required) and returns an already-complete request. This makes it
portable but means every rank participates in the exchange even if it has no
data to send. With the host executor the exchange does not overlap with
computation; on device executors the local-matrix kernel is queued before the
blocking exchange is issued, so that kernel does overlap (work issued after the
call does not).

Pick `DenseCommunicator` when in doubt; it works on every
communicator topology and performs well at small-to-moderate process
counts. Switch to [`NeighborhoodCommunicator`](neighborhood-communicator.md)
at large rank counts where each rank only communicates with a few
neighbours.

```{doxygenclass} gko::experimental::mpi::DenseCommunicator
:members:
```
