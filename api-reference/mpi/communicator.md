# `gko::experimental::mpi::communicator`

Owning C++ wrapper around `MPI_Comm`. Every distributed Ginkgo type
(`distributed::Matrix`, `distributed::Vector`,
`distributed::Partition` consumers, …) carries one of these. The
class is reference-counted, so passing it by value is cheap and the
underlying `MPI_Comm` is freed when the last `communicator` instance
goes out of scope.

Most application code constructs a `communicator` once around
`MPI_COMM_WORLD` (or a derived communicator) and threads it through
the distributed types it creates.

```cpp
gko::experimental::mpi::communicator comm{MPI_COMM_WORLD};
auto mat = gko::experimental::distributed::Matrix<...>::create(exec, comm);
```

The `rank()` and `size()` accessors return the current rank and
total process count without going through `MPI_Comm_rank` /
`MPI_Comm_size` on every call.

```{doxygenclass} gko::experimental::mpi::communicator
:members:
```
