# `gko::experimental::distributed::`<br>`DistributedBase`

Mixin that gives a distributed type access to its MPI communicator
through `get_communicator()`. `Matrix`, `Vector` and `RowGatherer`
inherit from it and expose the underlying `mpi::communicator` for
direct use in MPI-aware application code.

```{doxygenclass} gko::experimental::distributed::DistributedBase
:members:
```
