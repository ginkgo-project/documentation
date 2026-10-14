# `gko::experimental::distributed::`<br>`Partition`

Ownership map: stores, for each contiguous range of global indices,
which rank owns it. Three factory helpers cover the common cases:

- `build_from_contiguous`: explicit per-rank ranges.
- `build_from_global_size_uniform`: uniform block split of a given
  global size.
- `build_from_mapping`: generic per-row ownership array (e.g. the
  output of ParMETIS).

A `Matrix` takes its row partition, and optionally a different
column partition, in `read_distributed`.

```{doxygenclass} gko::experimental::distributed::Partition
:members:
```
