# `gko::experimental::mpi::contiguous_type`

Move-only RAII wrapper around a contiguous `MPI_Datatype`. The
wrapper builds the datatype via `MPI_Type_contiguous`, commits it,
and frees it on destruction.

```cpp
gko::experimental::mpi::contiguous_type t{3, MPI_DOUBLE};
// t.get() is a committed MPI_Datatype representing 3 doubles
```

Used internally by the distributed kernels to ship multi-vector
columns as a single typed message rather than as one `MPI_Send` per
column.

```{doxygenclass} gko::experimental::mpi::contiguous_type
:members:
```
