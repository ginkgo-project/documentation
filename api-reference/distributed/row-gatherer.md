# `gko::experimental::distributed::`<br>`RowGatherer`

Communication primitive that fetches remote rows of a distributed
vector — the workhorse behind the non-local SpMV in
`distributed::Matrix`. Caches the send and receive index lists so
repeated halo exchanges with the same partition reuse the same
schedule and only the payload moves.

`apply_async` writes into a distributed `Vector` whose local row count is
the collective communicator's `get_recv_size()`, although the parameter
documentation below names a `matrix::Dense` (a `matrix::Dense` output
throws `NotSupported`). The class example below likewise shows a
`matrix::Dense`; build `x` as a distributed `Vector` instead, as shown in
[RowGatherer](../../user-guide/concepts/distributed/row-gather-scatter.md).

```{doxygenclass} gko::experimental::distributed::RowGatherer
:members:
```
