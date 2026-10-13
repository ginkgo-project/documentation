# `gko::experimental::mpi::communicator`

Reference-counted C++ wrapper around `MPI_Comm`. The distributed
`Matrix`, `Vector` and `RowGatherer` each carry one of these, and passing
it by value is cheap. Wrapping an existing `MPI_Comm` leaves its ownership
with the application; the splitting constructors and `create_owning`
own the resulting `MPI_Comm`, which is freed when the last copy of the
`communicator` goes out of scope.

Most application code constructs a `communicator` once around
`MPI_COMM_WORLD` (or a derived communicator) and threads it through
the distributed types it creates.

```cpp
gko::experimental::mpi::communicator comm{MPI_COMM_WORLD};
int rank = comm.rank();
int size = comm.size();
auto exec = gko::ReferenceExecutor::create();
auto mat = gko::experimental::distributed::Matrix<
    double, gko::int32, gko::int64>::create(exec, comm);
```

The `rank()` and `size()` accessors query MPI (`MPI_Comm_rank` /
`MPI_Comm_size`) for the current rank and the total process count.

```{doxygenclass} gko::experimental::mpi::communicator
:members:
```
