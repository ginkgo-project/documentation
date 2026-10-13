# `gko::experimental::mpi::`<br>`DenseCommunicator`

Implementation of [`CollectiveCommunicator`](collective-communicator.md).
Exchanges with a blocking `MPI_Alltoallv` over the standard MPI communicator
(no graph topology required), and the request it returns is already complete.
This makes it portable, but every rank participates in the exchange even if it
has no data to send, and the exchange does not overlap with computation.

Ginkgo uses `DenseCommunicator` as its default collective communicator when it
is configured against Open MPI older than 4.1. Otherwise the default is
[`NeighborhoodCommunicator`](neighborhood-communicator.md).
`DenseCommunicator` works on every communicator topology and performs well at
small-to-moderate process counts. At large rank counts where each rank only
communicates with a few neighbours, `NeighborhoodCommunicator` is the better
fit.

```{doxygenclass} gko::experimental::mpi::DenseCommunicator
:members:
```
