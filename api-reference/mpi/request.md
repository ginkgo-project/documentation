# `gko::experimental::mpi::request`

Move-only handle representing an outstanding nonblocking MPI
operation. Returned by the asynchronous send/receive helpers on
`communicator` and by the nonblocking variants on the collective
communicators.

```cpp
auto req = comm.i_send(exec, buf, count, dest, tag);
// ... do other work ...
auto stat = req.wait();
```

`wait()` blocks until the operation completes and returns the
matching `status`. The `request` is consumed by `wait()`.

```{doxygenclass} gko::experimental::mpi::request
:members:
```
