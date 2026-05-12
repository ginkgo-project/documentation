# Partitions and index maps

`Partition` describes which global rows of a distributed matrix live on which MPI rank. `IndexMap`
resolves between local and global indexing. Together they are the bookkeeping layer that lets every
other distributed primitive work correctly. This page covers their construction patterns and the
data they expose.

## Partition

A `Partition<L, G>` is templated on a local index type (`L`) and a global index type (`G`). It maps
global row indices to ranks by storing ranges of contiguous global indices, each owned by one rank.
The partition is always constructed on a host executor — it is a host-side data structure used to
coordinate assembly and index translation.

Three construction patterns cover the common cases:

```cpp
namespace dist = gko::experimental::distributed;

// Contiguous: each rank owns a contiguous range of global rows.
//    `ranges_array` has length comm.size() + 1, with ranges_array[r] being the
//    first global row of rank r.
auto p1 = dist::Partition<gko::int32, gko::int64>::build_from_contiguous(
    host, ranges_array);

// Mapping: an external partitioner assigns each global row to a rank.
//    `mapping_array` has length == global_size; mapping_array[i] is the owner rank
//    of global row i.
auto p2 = dist::Partition<gko::int32, gko::int64>::build_from_mapping(
    host, mapping_array, n_parts);

// Uniform: Ginkgo divides N rows as evenly as possible among all ranks.
//    floor(N / size) rows per rank, with remainder distributed to the first few.
auto p3 = dist::Partition<gko::int32, gko::int64>::build_from_global_size_uniform(
    host, n_parts, global_size);
```

Use `build_from_contiguous` when your physical decomposition already gives contiguous row ranges
(common for structured grids and many FE meshes). Use `build_from_mapping` when an external
graph partitioner such as ParMETIS or Zoltan gives you a per-DOF rank assignment array. Use
`build_from_global_size_uniform` for quick experiments or for problems where no natural partition
exists.

## What Partition exposes

Once constructed, the partition exposes the range and size information for all ranks:

```cpp
auto bounds   = partition->get_range_bounds();   // array<G>: global start of each range
auto part_ids = partition->get_part_ids();        // array<comm_index_type>: owner of each range
auto sizes    = partition->get_part_sizes();      // array<L>: local row count per rank
auto n_parts  = partition->get_num_parts();       // number of MPI parts (== comm.size() typically)
auto n_ranges = partition->get_num_ranges();      // number of contiguous ranges (>= n_parts)
```

In the common case, `n_parts == comm.size()` and `n_ranges == n_parts` (one contiguous range per
rank). A mapping-based partition may produce more ranges than parts if the assignment is
non-contiguous.

## IndexMap

`IndexMap<L, G>` is the per-rank mapping between local indices and global indices. It lives
alongside a `Partition` and a communicator. The distributed `Matrix` and `Vector` use it to
translate between indexing schemes for the halo exchange — determining which remote global indices
this rank's off-diagonal matrix references, and which remote ranks need values from this rank.

For most user code, `IndexMap` is an implementation detail accessed indirectly:

```cpp
auto index_map = mat->get_index_map();   // shared_ptr<const IndexMap<L, G>>
```

Direct construction of an `IndexMap` is rarely needed. `read_distributed` derives the index map
automatically from the matrix's sparsity pattern, so you get a correctly-built map for free.

## Choosing a partition

A good partition minimises the off-diagonal-matrix size, which directly controls communication
volume. Practical guidelines:

- **Match your application's decomposition.** For FE codes, partition by element groups mapped to
  DOF groups, not by raw row indices. A uniform-row partition usually cuts across element
  boundaries, creating unnecessary off-diagonal entries.
- **Balance non-zero count, not just row count.** Skewed sparsity (some rows much denser than
  others) makes uniform-row partitions load-imbalanced for SpMV. If your matrix has strongly
  non-uniform row lengths, a graph-partitioner-based mapping handles this better.
- **Minimise off-diagonal entries.** Communication volume is proportional to the number of
  off-diagonal non-zeros summed across all ranks. Mesh-aware partitioners minimise the cut,
  which minimises the halo size.

:::{tip}
For a quick start or a test, `build_from_global_size_uniform` is the least code. For production
runs on unstructured meshes, invest in a graph-partitioner-based mapping — the convergence benefit
from a smaller off-diagonal matrix typically outweighs the setup cost.
:::

:::{seealso}
- [Distributed computing — overview](../distributed.md) — the assembly pattern that uses a
  `Partition`.
- [Communication primitives](communication-primitives.md) — what the index map drives during
  the distributed apply.
- API reference: [`gko::experimental::distributed::Partition`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1experimental_1_1distributed_1_1Partition.html)
:::
