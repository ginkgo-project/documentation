# Partitions and index maps

`Partition` describes which global rows of a distributed matrix live on which MPI rank. `index_map`
resolves between local and global indexing. Together they are the bookkeeping layer that lets every
other distributed primitive work correctly. This page covers their construction patterns and the
data they expose.

## Partition

A `Partition<L, G>` is templated on a local index type (`L`) and a global index type (`G`). It
maps global row indices to ranks by storing ranges of contiguous global indices, each owned by
one rank.

- **Executor support**: backend kernels exist for Reference, OMP, CUDA, HIP, and SYCL, so a
  partition built directly on a device executor is valid.
- **Cross-executor use**: `read_distributed` calls `make_temporary_clone(matrix_exec, partition)`
  internally, so any mismatch between the partition's executor and the matrix's is resolved
  transparently.
- **Where to construct**: the host is conventional for small partitions; constructing on the
  matrix's device avoids a host round-trip when the input arrays already live there.

Three construction patterns cover the common cases:

```cpp
namespace dist = gko::experimental::distributed;

// Contiguous: each rank owns a contiguous range of global rows.
//    `ranges_array` has length comm.size() + 1, with ranges_array[r] being the
//    first global row of rank r.
auto p1 = dist::Partition<gko::int32, gko::int64>::build_from_contiguous(
    exec, ranges_array);

// Mapping: an external partitioner assigns each global row to a rank.
//    `mapping_array` has length == global_size; mapping_array[i] is the owner rank
//    of global row i.
auto p2 = dist::Partition<gko::int32, gko::int64>::build_from_mapping(
    exec, mapping_array, n_parts);

// Uniform: Ginkgo divides N rows as evenly as possible among all ranks.
//    floor(N / size) rows per rank, with remainder distributed to the first few.
auto p3 = dist::Partition<gko::int32, gko::int64>::build_from_global_size_uniform(
    exec, n_parts, global_size);
```

Use `build_from_contiguous` when your physical decomposition already gives contiguous row ranges
(common for structured grids and many FE meshes). Use `build_from_mapping` when an external
graph partitioner such as ParMETIS or Zoltan gives you a per-DOF rank assignment array. Use
`build_from_global_size_uniform` for quick experiments or for problems where no natural partition
exists.

## What Partition exposes

Once constructed, the partition exposes the range and size information for all ranks:

```cpp
auto bounds   = partition->get_range_bounds();   // const G*: global start of each range (n_ranges + 1 entries)
auto part_ids = partition->get_part_ids();        // const comm_index_type*: owner of each range
auto sizes    = partition->get_part_sizes();      // const L*: local row count per part
auto n_parts  = partition->get_num_parts();       // number of MPI parts (== comm.size() typically)
auto n_ranges = partition->get_num_ranges();      // number of contiguous ranges (>= n_parts)
```

In the common case, `n_parts == comm.size()` and `n_ranges == n_parts` (one contiguous range per
rank). A mapping-based partition may produce more ranges than parts if the assignment is
non-contiguous.

## index_map

`index_map<L, G>` is the per-rank mapping between local indices and global indices. It lives
alongside a `Partition` and a communicator. The distributed `Matrix`, and the communicators and `RowGatherer` built from its index map, use it
to translate between indexing schemes for the halo exchange — determining which remote global indices
this rank's non-local matrix references, and which remote ranks need values from this rank.

For most user code, `index_map` is an implementation detail: `read_distributed` derives the map
from the matrix's sparsity pattern and the matrix keeps it internally, so you get a
correctly-built map for free. Constructing an `index_map` directly is needed only when building
a `RowGatherer` or a communicator by hand (see
[RowGatherer](row-gather-scatter.md)).

## Choosing a partition

A good partition minimises the non-local-matrix size, which directly controls communication
volume. Practical guidelines:

- **Match your application's decomposition.** For FE codes, partition by element groups mapped to
  DOF groups, not by raw row indices. A uniform-row partition usually cuts across element
  boundaries, creating unnecessary non-local entries.
- **Balance non-zero count, not just row count.** Skewed sparsity (some rows much denser than
  others) makes uniform-row partitions load-imbalanced for SpMV. If your matrix has strongly
  non-uniform row lengths, a graph-partitioner-based mapping handles this better.
- **Minimise non-local entries.** Communication volume is proportional to the number of
  non-local non-zeros summed across all ranks. Mesh-aware partitioners minimise the cut,
  which minimises the halo size.

:::{tip}
For a quick start or a test, `build_from_global_size_uniform` is the least code. For production
runs on unstructured meshes, invest in a graph-partitioner-based mapping — the convergence benefit
from a smaller non-local matrix typically outweighs the setup cost.
:::

:::{seealso}
- [Distributed computing — overview](../distributed.md) — the assembly pattern that uses a
  `Partition`.
- [Communication primitives](communication-primitives.md) — what the index map drives during
  the distributed apply.
- API reference: [`gko::experimental::distributed::Partition`](../../../api-reference/distributed/partition.md)
:::
