# `gko::experimental::distributed`

Distributed-memory linear algebra primitives. Every type here is
partitioned across MPI ranks: a `Partition` assigns ranges of global
rows to ranks, and off-rank columns are exchanged through halo
communication during `apply`. `Matrix` and `preconditioner::Schwarz` are templated on
`(ValueType, LocalIndexType, GlobalIndexType)`, `Vector` on `ValueType`,
`Partition` and `index_map` on the two index types, and `RowGatherer`
on `LocalIndexType`. The build must include MPI
(`GINKGO_BUILD_MPI`).

:::{attention}
Everything in `gko::experimental::distributed` is experimental: the
interfaces may change between releases.
:::

For background and a worked example see the
[Distributed concept page](../../user-guide/concepts/distributed.md).

## Distributed types

- [`Matrix`](matrix.md) — distributed sparse matrix, stored as an
  on-rank "diagonal" block plus a compressed "off-diagonal" block of
  remote columns.
- [`Vector`](vector.md) — distributed dense vector; each rank owns the
  rows that match the partition.
- [`Partition`](partition.md) — ownership mapping: which global rows /
  columns belong to which rank.
- [`index_map`](index-map.md) — translation between global and local
  indices for the partitioned space.
- [`RowGatherer`](row-gatherer.md) — communication primitive that
  fetches remote rows for the off-diagonal SpMV.
- [`DistributedBase`](distributed-base.md) — mixin that gives a type
  access to its MPI communicator.

## Preconditioners

- [`preconditioner::Schwarz`](schwarz.md) — additive Schwarz; applies
  any local solver per subdomain (rank), optionally with an L1
  smoother and an algebraic coarse-grid correction.

## Free functions

- [`assemble_rows_from_neighbors`](assembly.md) — sums up per-rank
  contributions to globally-owned rows so a `matrix_data` assembled
  independently on each rank can be used to build a distributed
  matrix.
- [`build_partition_from_local_range`, `build_partition_from_local_size`](partition.md)
  — collective helpers that build a `Partition` from each rank's own
  range or size.

```{toctree}
:hidden:
:maxdepth: 1

matrix
vector
partition
index-map
row-gatherer
distributed-base
schwarz
assembly
```
