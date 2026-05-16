# `gko::experimental::mpi::status`

Lightweight wrapper around `MPI_Status`. Returned by `request::wait`
and the synchronous receive helpers, and used to query the number of
elements actually received via `get_count<T>(data)`.

```{doxygenstruct} gko::experimental::mpi::status
:members:
```
