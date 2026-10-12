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

### Building from per-rank data

When every rank only knows its own share of the rows, two collective helpers build the partition
without gathering the sizes by hand. Both live in `gko::experimental::distributed` and are
collective over the communicator:

- `build_partition_from_local_size(exec, comm, local_size)` — rank $r$ owns `local_size`
  consecutive rows, and the ranges follow in rank order. The sizes may differ between ranks.
- `build_partition_from_local_range(exec, comm, local_range)` — each rank passes the
  `gko::span` of global rows it owns; the ranges may be given by the ranks in any order, as long
  as together they cover the index space without gaps (otherwise it throws [`gko::InvalidStateError`](../../../api-reference/base/exceptions.md)).

```cpp
// Each rank owns `local_size` consecutive rows, in rank order.
auto partition = gko::share(
    gko::experimental::distributed::build_partition_from_local_size<
        gko::int32, gko::int64>(exec, comm, local_size));
```

## What Partition exposes

Once constructed, the partition exposes the range and size information for all ranks:

```cpp
const gko::int64* bounds = partition->get_range_bounds();   // num_ranges + 1 entries: start of each range
const gko::experimental::mpi::comm_index_type* part_ids =
    partition->get_part_ids();                              // owner of each range
const gko::int32* sizes = partition->get_part_sizes();      // local row count per part
auto my_rows  = partition->get_part_size(comm.rank());      // rows owned by this rank
auto n_parts  = partition->get_num_parts();                 // number of MPI parts (== comm.size() typically)
auto n_ranges = partition->get_num_ranges();                // number of contiguous ranges (>= n_parts)
auto global_n = partition->get_size();                      // total number of indices
bool same     = partition->equals(*other_partition);        // same distribution?
```

The getters return pointers into the partition's own arrays, which live on the partition's
executor. `equals` checks that two partitions span the same global size and assign the same ranges
to the same parts; it is free for a partition compared with itself and copies the range data to the
host for two distinct device partitions. Distributed `multiply` uses it to check that the column
partition of `A` matches the row partition of `B`.

In the common case, `n_parts == comm.size()` and `n_ranges == n_parts` (one contiguous range per
rank). A mapping-based partition may produce more ranges than parts if the assignment is
non-contiguous.

## index_map

`index_map<L, G>` is the per-rank mapping between local indices and global indices. It holds the
partition it was built from (`get_partition()`) and this rank's id. The distributed `Matrix` uses it
to translate between indexing schemes for the halo exchange — determining which remote global
indices this rank's off-diagonal matrix references, and which remote ranks need values from this
rank.

`read_distributed` builds the `index_map` from the column partition and the off-diagonal column
indices, and the matrix keeps it internally. What user code can reach on the matrix is the result of
that bookkeeping:

```cpp
auto row_part = mat->get_row_partition();   // shared_ptr<const Partition<L, G>>, set by read_distributed
auto rg       = mat->get_row_gatherer();    // drives the halo exchange built from the index map
```

Construct one directly, `index_map<L, G>(exec, partition, rank, recv_connections)`, when building a
`RowGatherer` or a collective communicator yourself. `get_partition()`, `get_remote_target_ids()`,
`get_remote_global_idxs()` and `get_remote_local_idxs()` expose its contents.

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
- API reference: [`gko::experimental::distributed::Partition`](../../../api-reference/distributed/partition.md)
:::
