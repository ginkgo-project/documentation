# `gko::experimental::mpi::`<br>`DenseCommunicator`

Default implementation of
[`CollectiveCommunicator`](collective-communicator.md). Uses
`MPI_Ineighbor_alltoallv` over the standard MPI communicator (no
graph topology required), which makes it portable but means every
rank participates in the exchange even if it has no data to send.

Pick `DenseCommunicator` when in doubt — it works on every
communicator topology and performs well at small-to-moderate process
counts. Switch to [`NeighborhoodCommunicator`](neighborhood-communicator.md)
at large rank counts where each rank only communicates with a few
neighbours.

```{doxygenclass} gko::experimental::mpi::DenseCommunicator
:members:
```
