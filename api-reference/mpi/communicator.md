# `gko::experimental::mpi::communicator`

Reference-counted C++ wrapper around `MPI_Comm`. The distributed
`Matrix`, `Vector` and `RowGatherer` each carry one of these, and copies are
cheap. Wrapping an existing `MPI_Comm` leaves its ownership with the
application; communicators created by splitting
(`communicator(comm, color, key)`) own their `MPI_Comm` and free it when
the last copy goes out of scope.

Most application code constructs a `communicator` once around
`MPI_COMM_WORLD` (or a derived communicator) and threads it through
the distributed types it creates.

```cpp
gko::experimental::mpi::communicator comm{MPI_COMM_WORLD};
auto mat = gko::experimental::distributed::Matrix<double, gko::int32, gko::int64>::create(
    exec, comm);
```

`rank()`, `size()` and `node_local_rank()` query the rank, the process
count and the node-local rank.

```{doxygenclass} gko::experimental::mpi::communicator
:members:
```
