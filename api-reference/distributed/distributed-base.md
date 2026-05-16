# `gko::experimental::distributed::`<br>`DistributedBase`

Mixin that gives a distributed type access to its MPI communicator
through `get_communicator()`. Every type in this namespace inherits
from it, so any `Matrix`, `Vector`, `Partition`, `RowGatherer`, or
preconditioner exposes the underlying `mpi::communicator` for
direct use in MPI-aware application code.

```{doxygenclass} gko::experimental::distributed::DistributedBase
:members:
```
