# Distributed computing: overview

For multi-GPU and multi-node systems, Ginkgo provides distributed `Matrix` and `Vector` types under
`gko::experimental::distributed`. This page introduces the abstraction, how a distributed matrix is
assembled, when to choose distributed over single-device, the communicator, and the matrix-level
operations (`multiply`, `transpose`, `write`). Per-topic detail (partitioning, communication,
distributed-aware solvers) lives in the sub-pages.

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

- `Matrix<V, L, G>`: a distributed sparse matrix. `V` is the value type, `L` is the local index
  type (typically `gko::int32`), and `G` is the global index type (typically `gko::int64`).
- `Vector<V>`: a distributed dense vector or multi-vector.

Both inherit from `LinOp`, so they compose with solvers and preconditioners using exactly the same
patterns as single-device objects. A CG solver doesn't know whether the matrix it operates on is
distributed; it just calls `apply`.

:::{attention}
Everything in `gko::experimental::distributed` and `gko::experimental::mpi` is experimental: the
interfaces, including the matrix-level operations below, may change between releases.
:::

### Assembling the matrix

`read_distributed` runs on the executor the matrix was created on. Assembling on the host and then
cloning to the device (shown below) is the portable pattern and the one to use with
`DpcppExecutor`; with `OmpExecutor`, `CudaExecutor` and `HipExecutor` the matrix can also be
created directly on the compute executor, and `read_distributed` is called there:

```cpp
namespace dist = gko::experimental::distributed;

auto host = gko::ReferenceExecutor::create();
auto exec = gko::CudaExecutor::create(0, host);
auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);

// Each rank builds its local contribution to the global matrix.
gko::matrix_data<double, gko::int64> data;

// Build a Partition: which global rows live on which rank.
auto partition = gko::share(
    dist::Partition<gko::int32, gko::int64>::build_from_contiguous(
        host, ranges_array));

// Assemble the distributed matrix on the host.
auto mat_host = dist::Matrix<double, gko::int32, gko::int64>::create(host, comm);
mat_host->read_distributed(data, partition);

// Clone to the device for computation.
auto mat = mat_host->clone(exec);
```

`read_distributed` takes the partition as a `std::shared_ptr`, which is why the builder's result is
wrapped in `gko::share`. It coordinates the partition across ranks; once assembled, `clone(exec)`
copies the local blocks to the device in one step. Besides the one-partition overload shown above,
`read_distributed` accepts separate row and column partitions, and both overloads take either
`matrix_data` or `device_matrix_data`.

Distributed `Vector` assembly follows the same pattern:

```cpp
auto vec_host = dist::Vector<double>::create(host, comm, global_size, local_size);
// Fill local entries ...
auto vec = vec_host->clone(exec);
```

### The communicator

The distributed `Matrix`, `Vector` and `RowGatherer` each hold a `gko::experimental::mpi::communicator`:

```cpp
auto comm = mat->get_communicator();
int rank  = comm.rank();
int size  = comm.size();
MPI_Comm raw = comm.get();   // for direct MPI use if needed
```

Rank 0 is the leader by convention. Each rank owns only its local part of the matrix and vector
data. Global operations (dot products, norms) reduce across ranks internally using `MPI_Allreduce`.

## Block-structured view across ranks

Conceptually, the global matrix is permuted by the row partition so that the rows owned by each
rank form a contiguous block. Within that block-row, the columns split by ownership (per the
column partition, by default the same as the row partition). This gives every rank two pieces
of the matrix:

- **Diagonal matrix**: entries whose row is owned by this rank **and** whose column is also
  owned by this rank. Across all ranks, these pieces sit on the block-diagonal of the permuted
  global matrix; hence the name. Both row and column indices are local (0..*n_local* − 1) and
  the dimensions are $n_\text{local rows} \times n_\text{local cols}$. This is the bulk of the
  work for well-partitioned matrices.
- **Off-diagonal matrix**: entries whose row is owned by this rank but whose column is *not*
  owned by this rank. These sit on the off-diagonal blocks of the permuted global matrix and
  drive the halo exchange during `apply`. The storage is **compressed**: only the remote
  columns that this rank's local rows actually reference are kept, and those columns are
  renumbered into a compact $0..K-1$ local indexing via the matrix's `index_map`. So its
  dimensions are $n_\text{local rows} \times K$, where $K$ is the number of **distinct** remote
  global columns this rank touches, **not** the total number of remote columns. The original
  global column IDs are tracked separately in `index_map::get_remote_global_idxs()` and used
  to unpack the values received in the halo exchange.

:::{note}
"Diagonal" here refers to the **block structure** of the partitioned global matrix, not to the
diagonal of the matrix in the linear-algebra sense. The diagonal matrix on each rank is an
arbitrary sparse sub-block of $A$ that this rank owns end-to-end. The off-diagonal matrix is
similarly a sparse block; the "off-diagonal" name describes its position in the partitioned
global matrix, not anything about its values.
:::

### A worked example

Take a 4 × 4 matrix on 2 ranks. Two partitions are involved, and they can be different
functions, though by default Ginkgo uses the row partition for both:

- **Row partition**: assigns each global row index to one rank.
- **Column partition**: assigns each global column index to one rank.

Saying "rank $R$ owns row $i$" means the row partition maps $i$ to $R$; saying "rank $R$ owns
column $j$" means the column partition maps $j$ to $R$. Ownership is a property of the
partition, not of which rank physically stores something; those are related but distinct
questions, and we will see how, below.

For this example, let both partitions assign indices {0, 1} to rank 0 and {2, 3} to rank 1.
Suppose the global sparsity pattern is:

```text
                       column partition
                ┌── cols owned by ──┬── cols owned by ──┐
                │      rank 0       │      rank 1       │
                │  col 0    col 1   │  col 2    col 3   │
                ├───────────────────┼───────────────────┤
                │                   │                   │
   row 0  ───┐  │   *        *      │   *        .      │
   row 1     │  │   *        *      │   *        .      │   ← rows owned by rank 0
             │  │                   │                   │
             │  ├───────────────────┼───────────────────┤
             │  │                   │                   │
   row 2     │  │   .        .      │   *        *      │   ← rows owned by rank 1
   row 3  ───┘  │   *        .      │   *        *      │
                └───────────────────┴───────────────────┘
                  ↑                    ↑
                  │                    │
            cols 0,1 belong to    cols 2,3 belong to
            rank 0 (per the       rank 1 (per the
            column partition,     column partition,
            globally)             globally)
```

The four quadrants of the matrix are defined by **crossing** row ownership with column
ownership. Each quadrant has a name based on the row owner *and* on whether the column owner
matches:

```text
                       cols owned by R0     cols owned by R1
                       ┌──────────────────┬──────────────────┐
   rows owned by R0    │   R0's diagonal  │ R0's off-diagonal│   ←─ stored on rank 0
                       │  (R0 owns both   │ (R0 owns rows,   │
                       │   rows and cols) │  R1 owns cols)   │
                       ├──────────────────┼──────────────────┤
   rows owned by R1    │R1's off-diagonal │  R1's diagonal   │   ←─ stored on rank 1
                       │ (R1 owns rows,   │ (R1 owns both    │
                       │  R0 owns cols)   │  rows and cols)  │
                       └──────────────────┴──────────────────┘
```

Where each block is **stored** is determined by the *row* owner alone: every rank holds the
two blocks of the global matrix that lie in its own row range. So:

- **Rank 0** stores its top-left quadrant (its diagonal matrix) and its top-right quadrant
  (its off-diagonal matrix). The cols 2, 3 block lives on rank 0 because rows 0, 1 live on
  rank 0, even though rank 1 is the one that *owns* those columns.
- **Rank 1** stores the bottom-left quadrant (its off-diagonal matrix) and the bottom-right
  quadrant (its diagonal matrix). Symmetric story.

Plugging in the concrete sparsity pattern above, the per-rank storage is:

```text
Rank 0 (holds rows 0, 1):
                                          index_map for the off-diagonal cols:
  diagonal matrix      off-diagonal         local col 0  ↔  global col 2
  (2 × 2)              matrix (2 × K)       (col 3 is not touched — dropped)
  ┌────────────┐       ┌─────┐
  │  *    *    │       │  *  │              K = 1: only one of the two remote
  │  *    *    │       │  *  │              columns is actually referenced.
  └────────────┘       └─────┘

Rank 1 (holds rows 2, 3):
                                          index_map for the off-diagonal cols:
  diagonal matrix      off-diagonal         local col 0  ↔  global col 0
  (2 × 2)              matrix (2 × K)       (col 1 is not touched — dropped)
  ┌────────────┐       ┌─────┐
  │  *    *    │       │  .  │              K = 1: again, only one of the two
  │  *    *    │       │  *  │              remote columns is referenced.
  └────────────┘       └─────┘
```

Note that `get_off_diag_matrix()` on rank 0 returns a 2 × 1 sparse block, not a 2 × 2 block:
the remote column the rank doesn't touch (global col 3) is simply not stored. The mapping
from the compressed local column to its global index lives in
`index_map::get_remote_global_idxs()` and is used during the halo exchange to know which
entries of $x$ this rank should expect from which other rank.

Access the parts directly if needed:

```cpp
auto diag_mat     = mat->get_diag_matrix();       // shared_ptr<const LinOp>: the diagonal matrix
auto off_diag_mat = mat->get_off_diag_matrix();   // the off-diagonal matrix (compressed columns)
auto row_part     = mat->get_row_partition();     // the row partition passed to read_distributed
auto row_gatherer = mat->get_row_gatherer();      // drives the halo exchange of apply

// Vectors expose the rank-local Dense:
auto local_x = x->get_local_vector();             // matrix::Dense<V> with rank-owned rows only
```

`get_row_partition()` returns the partition the matrix was filled with by `read_distributed`;
`get_row_gatherer()` returns the `RowGatherer` whose collective communicator determines how the
matrix exchanges its halo data (see [Collective communicators](distributed/collective-communicator.md)).

To reach the rank-local block of a vector that may or may not be distributed, check its type:

```cpp
const gko::matrix::Dense<double>* local =
    dynamic_cast<const dist::Vector<double>*>(ptr)
        ? gko::as<dist::Vector<double>>(ptr)->get_local_vector()
        : gko::as<gko::matrix::Dense<double>>(ptr);
```

## Multiplying distributed matrices

`A->multiply(B, C)` computes the sparse product $C = A B$ across ranks. `C` is a distributed
matrix created beforehand on the same communicator; it may be empty, and any previous contents are
overwritten. The result has the row partition of `A` and the column partition of `B`, so it can be
the operand of a further `multiply`, as in a Galerkin product $R A P$. The snippet shows the
matrix-level operations together; the next two sections cover `transpose` and `write`.

```cpp
// C = A * B, with A, B and C distributed over the same communicator.
auto C = dist_mtx::create(exec, comm);
A->multiply(B, C);

// At = A^T, distributed like the columns of A.
auto At = dist_mtx::create(exec, comm);
A->transpose(At);

// The entries owned by this rank, with global row and column indices.
gko::matrix_data<double, gko::int64> local_entries;
A->write(local_entries);
```

`multiply` has these preconditions:

- The column partition of `A` equals the row partition of `B` (`Partition::equals` is the check);
  otherwise it throws [`gko::InvalidStateError`](../../api-reference/base/exceptions.md).
- Both operands have their partitions set, by `read_distributed` or as the output of `multiply` or
  `transpose`.
- The diagonal and off-diagonal blocks of both operands are `Csr`, which is the default storage.
- 64-bit local indices are rejected with `gko::NotSupported` on `HipExecutor`, and on `CudaExecutor`
  before CUDA 13; every rank throws before any communication starts.

Internally the product ships whole rows of `B` along the halo pattern of `A`, reusing the row
gatherer's pattern through `CollectiveCommunicator::resize`.

## Transposing

`A->transpose(At)` fills `At` with $A^T$. The result takes the column partition of `A` as its row
partition and the row partition of `A` as its column partition, which makes it a valid operand of
`multiply` together with `A`. Entries whose transposed row is owned by another rank
are sent to that rank, so every rank calls `transpose` together (it is a collective call).

`A` must have been filled by `read_distributed`, and `At` must use the same executor and
communicator as `A`. Like `C` above, `At` is overwritten and may be empty.

## Writing the local entries

`A->write(data)` collects the entries stored on this rank into a `matrix_data` with global row and
column indices; the size recorded in `data` is the global size of `A`. `Matrix` implements
`WritableToMatrixData` for this, so the same call works through the generic interface. It covers
square matrices whose row and column partitions are the same, which is the case for the
single-partition `read_distributed`.

The sub-pages below cover the distributed machinery in depth.

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
- [Partitions and index maps](distributed/partition-and-index-map.md): how rows are assigned to
  ranks and how local/global indexing is reconciled.
- [Communication primitives](distributed/communication-primitives.md): halo exchange, stream-MPI
  ordering, and the diagonal / off-diagonal decomposition of SpMV.
- [The MPI layer](distributed/mpi-layer.md): the communicator object, supported MPI calls, and the
  collective communicator that backs the all-to-all exchange.
- [Collective communicators](distributed/collective-communicator.md): the
  `CollectiveCommunicator` interface and its dense / neighborhood implementations, including
  `resize`, which distributed `multiply` uses.
- [RowGatherer](distributed/row-gather-scatter.md): the helper that drives the
  halo exchange on top of a collective communicator.
- [Distributed solvers and preconditioners](distributed/solvers-and-preconditioners.md): which
  solvers work automatically and how to configure the Schwarz preconditioner.
- [The Executor model](executor-model.md): distributed objects still bind to an executor per rank.
- [Matrix data and assembly](matrix-data.md): `matrix_data` for host-side assembly.
- API reference: [`gko::experimental::distributed`](../../api-reference/distributed/index.md)
:::
