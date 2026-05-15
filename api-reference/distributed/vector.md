# `gko::experimental::distributed::Vector`

Distributed dense vector. Each rank owns the rows of the global
vector that match its partition; per-rank storage is a regular
`matrix::Dense`, accessible through `get_local_vector()`. Inner
products, norms, and other reductions perform a single MPI
`Allreduce` across the communicator.

```{doxygenclass} gko::experimental::distributed::Vector
:members:
```
