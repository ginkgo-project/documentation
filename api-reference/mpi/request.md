# `gko::experimental::mpi::request`

Move-only handle representing an outstanding nonblocking MPI
operation. Returned by the asynchronous send/receive helpers on
`communicator` and by the nonblocking variants on the collective
communicators.

```cpp
auto exec = gko::ReferenceExecutor::create();
std::vector<double> send_buf(100, 1.0), recv_buf(100);
int next = (comm.rank() + 1) % comm.size();
int prev = (comm.rank() + comm.size() - 1) % comm.size();
auto send_req = comm.i_send(exec, send_buf.data(), 100, next, 0);
auto recv_req = comm.i_recv(exec, recv_buf.data(), 100, prev, 0);
// ... do other work ...
auto stat = recv_req.wait();
send_req.wait();
```

`wait()` blocks until the operation completes and returns the
matching `status`. The `request` is consumed by `wait()`.

```{doxygenclass} gko::experimental::mpi::request
:members:
```
