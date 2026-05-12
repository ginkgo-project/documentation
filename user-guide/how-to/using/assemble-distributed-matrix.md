# Assemble a distributed matrix

`gko::experimental::distributed::Matrix` represents a sparse matrix
whose rows are partitioned across MPI ranks. Each rank owns a
contiguous block of global rows; entries with column indices outside
the rank's row range become *off-diagonal matrix* entries that
participate in the halo exchange during `apply`.

## The recipe

```cpp
#include <ginkgo/ginkgo.hpp>

using LocalIdx  = int;
using GlobalIdx = long;
using Matrix    = gko::experimental::distributed::Matrix<double, LocalIdx, GlobalIdx>;
using Partition = gko::experimental::distributed::Partition<LocalIdx, GlobalIdx>;

auto comm = gko::experimental::mpi::communicator{MPI_COMM_WORLD};
auto exec = gko::OmpExecutor::create();
auto host = exec->get_master();

// Decide ownership: which global rows each rank owns.
const GlobalIdx global_n = 1'000'000;
auto partition = Partition::build_from_global_size_uniform(
    host, comm.size(), global_n);

// Build the full matrix data on the host. Each rank can contribute
//    nonzeros for any global row; read_distributed routes them by ownership.
gko::matrix_data<double, GlobalIdx> data{gko::dim<2>{global_n, global_n}};
for (auto [i, j, v] : your_triplets_anywhere) {
    data.nonzeros.emplace_back(i, j, v);
}

// Build the distributed matrix.
auto A = gko::share(Matrix::create(exec, comm));
A->read_distributed(data, partition.get());
```

After `read_distributed`, each rank holds only the rows assigned to it
by `partition`. Within that block-row, the matrix is split by column
ownership into a *diagonal matrix* (this rank also owns the column) and
an *off-diagonal matrix* (the column is owned by some other rank).
A worked 4 × 4 example with the resulting per-rank storage is in
[Distributed Computing — Block-structured view across
ranks](../../concepts/distributed.md#a-worked-example); the short
version:

```text
Rank R's share of A:

       cols R owns        cols other ranks own
       ┌───────────────┬──────────────────────┐
       │  diagonal     │  off-diagonal        │
       │  matrix       │  matrix (compressed) │
       └───────────────┴──────────────────────┘
```

The off-diagonal matrix is stored compressed: it keeps only the remote
columns this rank's rows actually reference, renumbered into a compact
$0..K-1$ local indexing via the matrix's `index_map`. "Diagonal"
here refers to the block structure of the partitioned global matrix,
not to the diagonal of the matrix in the linear-algebra sense.

## Pick a partition

| Builder | Use when |
|---|---|
| `Partition::build_from_global_size_uniform(exec, num_parts, global_n)` | Rows have no strong locality — short-range or no remote coupling — so a uniform block partition is good enough. |
| `Partition::build_from_contiguous(exec, ranges)` | You want explicit per-rank row ranges (e.g. load-balanced manually, or matching an existing application-side decomposition). |
| `Partition::build_from_mapping(exec, mapping, num_parts)` | You have a per-row owner mapping from an external graph partitioner (e.g. ParMETIS / Scotch output). |

`Partition` works on any executor — backend kernels exist for Reference, OMP, CUDA, HIP, and
SYCL. Building on the host is the conventional choice for small partitions; building on the
matrix's device avoids a host round-trip if the input arrays already live there. When passed to
`read_distributed`, the matrix transparently clones the partition to its own executor through
`make_temporary_clone` if the two executors differ.

## Build the matching vectors

Right-hand sides and solution vectors must use the same partition:

```cpp
using Vector = gko::experimental::distributed::Vector<double>;
auto b = Vector::create(exec, comm,
                        gko::dim<2>{global_n, 1},
                        gko::dim<2>{partition->get_part_sizes()[comm.rank()], 1});
auto x = gko::clone(exec, b);
// Fill b->get_local_vector()->at(i, 0) on each rank for the rank-local rows.
```

The two `dim<2>` arguments are the *global* size and the *local* size
respectively.

## Solve

Distributed-ready Krylov solvers (`Cg`, `Fcg`, `Gmres`, `Bicgstab`,
`Ir`, `Chebyshev`, `Multigrid` — see
[Distributed solvers and preconditioners](../../concepts/distributed/solvers-and-preconditioners.md))
work without modification:

```cpp
auto solver = gko::solver::Cg<double>::build()
    .with_criteria(
        gko::stop::Iteration::build().with_max_iters(1000u).on(exec),
        gko::stop::ResidualNorm<double>::build()
            .with_reduction_factor(1e-8).on(exec))
    .with_preconditioner(
        gko::experimental::distributed::preconditioner::Schwarz<double, LocalIdx, GlobalIdx>::build()
            .with_local_solver(/* per-rank preconditioner factory */)
            .on(exec))
    .on(exec)
    ->generate(A);
solver->apply(b, x);
```

## Common pitfalls

- **`matrix_data` is host-only.** It's a host-side incremental-assembly
  type. The `Partition`, by contrast, is fine on any executor —
  `read_distributed` clones it to the matrix's executor automatically
  if needed.
- **Use a 64-bit global index type** (`long` / `int64`) when
  `global_n` exceeds the 32-bit range. The local index type can stay
  32-bit; only global indexing across ranks needs the wider type.
- **Partition mismatches.** The matrix and every vector apply'd to it
  must share the same partition. Constructing two vectors with
  different partitions and applying one through the other's solver
  silently produces wrong answers.
- **`MPI_Init` first.** Ginkgo's distributed types require MPI to be
  initialised — either call `MPI_Init` yourself or use
  `gko::experimental::mpi::environment` (RAII).

:::{seealso}
- [Use an existing MPI communicator](use-existing-mpi-communicator.md) — wrap an `MPI_Comm` rather than using `MPI_COMM_WORLD`.
- [Distributed Ginkgo](../../concepts/distributed.md) — the conceptual reference.
- [Partition and IndexMap](../../concepts/distributed/partition-and-index-map.md) — partition options in detail.
:::
