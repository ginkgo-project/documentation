# `gko::experimental::distributed::RowGatherer`

Communication primitive that fetches remote rows of a distributed
vector — the workhorse behind the off-diagonal SpMV in
`distributed::Matrix`. Caches the send and receive index lists so
repeated halo exchanges with the same partition reuse the same
schedule and only the payload moves.

```{doxygenclass} gko::experimental::distributed::RowGatherer
:members:
```
