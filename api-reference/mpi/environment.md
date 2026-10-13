# `gko::experimental::mpi::environment`

RAII wrapper around `MPI_Init_thread` / `MPI_Finalize`. Construct one
at the top of `main` and let its destructor finalise MPI when the
program exits.

```cpp
int main(int argc, char* argv[]) {
    gko::experimental::mpi::environment env{argc, argv};
    // ... use Ginkgo distributed types ...
}
```

The constructor calls `MPI_Init_thread` unconditionally and requests a
thread-support level (`serialized` by default), so it is meant for
applications that have not initialised MPI yet. `is_initialized` and
`is_finalized` are static helpers that report whether `MPI_Init` and
`MPI_Finalize` have run.

```{doxygenclass} gko::experimental::mpi::environment
:members:
```
