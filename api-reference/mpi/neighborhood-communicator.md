# `gko::experimental::mpi::`<br>`NeighborhoodCommunicator`

Implementation of
[`CollectiveCommunicator`](collective-communicator.md) that uses
MPI-3 graph-topology neighborhood collectives
(`MPI_Ineighbor_alltoallv`). Each rank registers only the neighbours
it actually exchanges data with, and the MPI runtime is free to
schedule the communication accordingly.

`NeighborhoodCommunicator` is the collective communicator the distributed
`Matrix` uses by default. When Ginkgo is configured against Open MPI older than
4.1 the default is `DenseCommunicator` instead. `NeighborhoodCommunicator` suits large rank counts
where the communication graph is sparse, as is typical of distributed PDE
discretisations, where each rank only talks to its geometric
neighbours. On dense graphs or modest process counts,
[`DenseCommunicator`](dense-communicator.md) is simpler and often
just as fast.

```{doxygenclass} gko::experimental::mpi::NeighborhoodCommunicator
:members:
```
