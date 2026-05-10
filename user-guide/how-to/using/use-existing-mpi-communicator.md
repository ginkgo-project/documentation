# Use an existing MPI communicator

:::{note}
Stub — content forthcoming.
:::

Make Ginkgo's distributed types reuse an `MPI_Comm` your application already
created (sub-communicator, split, dup) instead of `MPI_COMM_WORLD`. Covers
constructing `gko::experimental::mpi::communicator` from a raw handle and the
ownership / lifetime guarantees Ginkgo makes around the wrapper.
