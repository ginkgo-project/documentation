# Assemble a distributed matrix

`gko::experimental::distributed::Matrix` represents a sparse matrix
whose rows are partitioned across MPI ranks. Each rank owns a
contiguous block of global rows; entries with column indices outside
the rank's row range become *ghost matrix* entries that participate in
the halo exchange during `apply`.

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
by `partition`, split into a *local matrix* (columns owned by this
rank) and a *ghost matrix* (columns owned by remote ranks).

## Pick a partition

| Builder | Use when |
|---|---|
| `Partition::build_from_global_size_uniform(host, num_parts, global_n)` | You don't care how rows are distributed; uniform block partition. |
| `Partition::build_from_contiguous(host, ranges)` | You want explicit per-rank row ranges (e.g. load-balanced manually). |
| `Partition::build_from_mapping(host, mapping, num_parts)` | You have a per-row owner mapping (e.g. ParMETIS output). |

`Partition` is host-resident; pass `host` as the executor.

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
`Ir`, `Minres`, `Multigrid`, etc. — see
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

- **Build `matrix_data` and `Partition` on the host.** `read_distributed`
  internally moves data onto the matrix's executor — passing a
  device-resident `Partition` is undefined behaviour.
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
