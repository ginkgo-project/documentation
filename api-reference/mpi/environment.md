# `gko::experimental::mpi::environment`

RAII wrapper around `MPI_Init_thread` / `MPI_Finalize`, for
applications that have not called `MPI_Init` themselves. Construct one
at the top of `main` and let its destructor finalise MPI when the
program exits.

```cpp
int main(int argc, char* argv[]) {
    gko::experimental::mpi::environment env{argc, argv};
    // ... use Ginkgo distributed types ...
}
```

The constructor requests a thread-support level (`serialized` by
default) and calls `MPI_Init_thread` unconditionally, so the guard is
meant for applications that have not initialised MPI themselves.
`is_initialized` and `is_finalized` are static helpers that report
whether `MPI_Init` and `MPI_Finalize` have run.

```{doxygenclass} gko::experimental::mpi::environment
:members:
```
