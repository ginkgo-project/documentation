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

The constructor requests a thread-support level (`serialized` by
default). `is_initialized` and `is_finalized` are static helpers that
let library code skip its own `MPI_Init` if the host application has
already initialised MPI.

```{doxygenclass} gko::experimental::mpi::environment
:members:
```
