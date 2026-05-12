# Distributed computing — overview

For multi-GPU and multi-node systems, Ginkgo provides distributed `Matrix` and `Vector` types under
`gko::experimental::distributed`. This page introduces the abstraction, the host-first assembly
pattern, when to choose distributed over single-device, and the communicator. Per-topic detail
(partitioning, communication, distributed-aware solvers) lives in the sub-pages.

## When to go distributed

Distributed is the right choice when a single device's memory is no longer sufficient or when a
multi-node MPI job demands a single matrix view spanning ranks:

- **Memory pressure.** A GPU typically becomes the bottleneck around $10^7$–$10^8$ unknowns. Once the
  matrix and vectors no longer fit in one device's VRAM, distributing across multiple GPUs (or CPU
  ranks) is the natural next step.
- **Multi-node MPI runs.** If your simulation already uses MPI for domain decomposition, Ginkgo's
  distributed types let you express the linear algebra at the same granularity as your physics.
- **Single-device otherwise.** If the problem fits comfortably on one GPU, stay single-device.
  Distributed is more code, introduces communication overhead, and has more failure modes.

## The distributed types

Two primary types live under `gko::experimental::distributed`:

- `Matrix<V, L, G>` — a distributed sparse matrix. `V` is the value type, `L` is the local index
  type (typically `gko::int32`), and `G` is the global index type (typically `gko::int64`).
- `Vector<V>` — a distributed dense vector or multi-vector.

Both inherit from `LinOp`, so they compose with solvers and preconditioners using exactly the same
patterns as single-device objects. A CG solver doesn't know whether the matrix it operates on is
distributed — it just calls `apply`.

## The host-first assembly pattern

Construction is always host-first, then transferred to the device executor. The `read_distributed`
step must run on a host executor because partition reasoning is host-side. Cloning to device after
assembly is the standard pattern:

```cpp
namespace dist = gko::experimental::distributed;

auto host = gko::ReferenceExecutor::create();
auto exec = gko::CudaExecutor::create(0, host);
auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);

// Each rank builds its local contribution to the global matrix.
gko::matrix_data<double, gko::int64> data{...};

// Build a Partition: which global rows live on which rank.
auto partition = dist::Partition<gko::int32, gko::int64>::build_from_contiguous(
    host, ranges_array);

// Assemble the distributed matrix on the host.
auto mat_host = dist::Matrix<double, gko::int32, gko::int64>::create(host, comm);
mat_host->read_distributed(data, partition.get());

// Clone to the device for computation.
auto mat = mat_host->clone(exec);
```

`read_distributed` inspects the global sparsity pattern across all ranks — it needs host-side MPI
to coordinate. Once assembled, `clone(exec)` copies the local blocks to the device in one step.

Distributed `Vector` assembly follows the same pattern:

```cpp
auto vec_host = dist::Vector<double>::create(host, comm, global_size, local_size);
// Fill local entries ...
auto vec = vec_host->clone(exec);
```

## The communicator

Every distributed object holds a `gko::experimental::mpi::communicator`:

```cpp
auto comm = mat->get_communicator();
int rank  = comm.rank();
int size  = comm.size();
MPI_Comm raw = comm.get();   // for direct MPI use if needed
```

Rank 0 is the leader by convention. No rank holds global state — each rank owns only its local
part of the matrix and vector data. Global operations (dot products, norms) reduce across ranks
internally using `MPI_Allreduce`.

## Block-structured view across ranks

Conceptually, the global matrix is permuted by the row partition so that the rows owned by each
rank form a contiguous block. Within that block-row, the columns split by ownership (per the
column partition — by default the same as the row partition). This gives every rank two pieces
of the matrix:

- **Diagonal matrix** — entries whose row is owned by this rank **and** whose column is also
  owned by this rank. Across all ranks, these pieces sit on the block-diagonal of the permuted
  global matrix; hence the name. Both row and column indices are local (0..*n_local* − 1) and
  the dimensions are $n_\text{local rows} \times n_\text{local cols}$. This is the bulk of the
  work for well-partitioned matrices.
- **Off-diagonal matrix** — entries whose row is owned by this rank but whose column is *not*
  owned by this rank. These sit on the off-diagonal blocks of the permuted global matrix and
  drive the halo exchange during `apply`. The storage is **compressed**: only the remote
  columns that this rank's local rows actually reference are kept, and those columns are
  renumbered into a compact 0..*K* − 1 local indexing via the matrix's `index_map`. So its
  dimensions are $n_\text{local rows} \times K$, where *K* is the number of *distinct* remote
  global columns this rank touches — **not** the total number of remote columns. The original
  global column IDs are tracked separately in `index_map::get_remote_global_idxs()` and used
  to unpack the values received in the halo exchange.

:::{note}
"Diagonal" here refers to the **block structure** of the partitioned global matrix, not to the
diagonal of the matrix in the linear-algebra sense. The diagonal matrix on each rank is an
arbitrary sparse sub-block of $A$ that this rank owns end-to-end. The off-diagonal matrix is
similarly a sparse block; the "off-diagonal" name describes its position in the partitioned
global matrix, not anything about its values.
:::

Access the parts directly if needed:

```cpp
auto diag_mat     = mat->get_local_matrix();      // shared_ptr<const LinOp>: the diagonal matrix
auto off_diag_mat = mat->get_non_local_matrix();  // the off-diagonal matrix (compressed columns)

// Vectors expose the rank-local Dense:
auto local_x = x->get_local_vector();             // matrix::Dense<V> with rank-owned rows only
```

The C++ API still spells these `get_local_matrix()` and `get_non_local_matrix()` for historical
reasons; the docs use *diagonal matrix* and *off-diagonal matrix* throughout, matching the
block-partitioning convention shared with PETSc, Trilinos, and the broader
distributed-sparse-linear-algebra literature.

For generic code that works for both distributed and non-distributed vectors, use the helper:

```cpp
gko::detail::get_local(ptr)   // returns Dense<V> whether ptr is dist::Vector or matrix::Dense
```

## Forward links

The sub-pages cover the distributed machinery in depth:

- [Partitions and index maps](distributed/partition-and-index-map.md) — how rows are assigned to
  ranks and how local/global indexing is reconciled.
- [Communication primitives](distributed/communication-primitives.md) — halo exchange, stream-MPI
  ordering, and the diagonal / off-diagonal decomposition of SpMV.
- [The MPI layer](distributed/mpi-layer.md) — the communicator object, supported MPI calls, and the
  collective communicator that backs the all-to-all exchange.
- [Collective communicators](distributed/collective-communicator.md) — the
  `CollectiveCommunicator` interface and its dense / neighborhood implementations.
- [RowGatherer and RowScatterer](distributed/row-gather-scatter.md) — the helper that drives the
  halo exchange on top of a collective communicator.
- [Distributed solvers and preconditioners](distributed/solvers-and-preconditioners.md) — which
  solvers work automatically and how to configure the Schwarz preconditioner.

## Distributed sub-pages

```{toctree}
:maxdepth: 2
:caption: Distributed

distributed/partition-and-index-map
distributed/communication-primitives
distributed/mpi-layer
distributed/collective-communicator
distributed/row-gather-scatter
distributed/solvers-and-preconditioners
```

:::{seealso}
- [The Executor model](executor-model.md) — distributed objects still bind to an executor per rank.
- [Matrix data and assembly](matrix-data.md) — `matrix_data` for host-side assembly.
- API reference: [`gko::experimental::distributed`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/namespacegko_1_1experimental_1_1distributed.html)
:::
