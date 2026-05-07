# Batched solvers — overview

For workloads that need to solve many small independent linear systems — flame chemistry,
multiphase Lagrangian particles, parameter sweeps — Ginkgo provides a batched subsystem.
All systems are solved simultaneously inside a single fused GPU kernel, with no inter-batch
communication. This page covers when batched is the right tool, the uniform-batch constraint,
the storage layout, and the fused-kernel design.

## When to use batched

Batched solvers are suited to:

- Many systems (`> 100`) of small size (typically `≤ 10⁴` rows each).
- All systems are independent — no coupling between batch items.
- Same matrix dimensions and same sparsity pattern across items (the uniform-batch constraint).

If those conditions hold, batched almost always outperforms looping over single-system solves:
kernel launch overhead is paid once instead of N times, and memory access can be coalesced
across batch items.

## The uniform-batch constraint

:::{attention}
All items in a batch must share:

- the same matrix dimensions (`num_rows`, `num_cols`),
- the same sparsity pattern — col indices and row pointers are stored **once**, shared across the
  batch,
- the same number of non-zeros per item.
:::

This is restrictive but it's what enables the fused-kernel design.

## Storage layout

For batched CSR, two arrays are batch-shared and one is per-item:

- `col_idxs` — single shared copy, length `nnz`.
- `row_ptrs` — single shared copy, length `num_rows + 1`.
- `values`   — per-item, total length `batch_items * nnz`. Item `k`'s values start at offset
  `k * nnz`.

```text
Batched CSR with 3 items (item k of N):

col_idxs:  [shared]                          length = nnz
row_ptrs:  [shared]                          length = num_rows + 1
values:    [item 0 | item 1 | item 2]        length = 3 * nnz
                                             item k at offset k * nnz
```

For batched dense, values are simply concatenated row-major per item.

## The fused-kernel design

The defining feature of batched solvers: the entire iterative solve lives inside one `__global__`
kernel.

- One CUDA block per batch item.
- Threads within a block cooperate via `__syncthreads()`.
- The iteration loop runs **inside** the kernel — no per-iteration kernel launch.
- Scalars (rho, alpha, residual norms) live in shared memory.
- Vectors (residual, search direction) are split between shared and global memory based on a
  `storage_config` heuristic that balances against available shared memory.

Why this matters: kernel launch overhead disappears, and shared-memory residency means
inner-loop reductions are fast.

## What's available

Matrix formats:

- `batch::matrix::Dense`
- `batch::matrix::Csr`
- `batch::matrix::Ell`
- `batch::matrix::Identity` — for the no-matrix (identity preconditioner) case.

Solvers:

- `batch::solver::Cg` — for symmetric positive-definite systems.
- `batch::solver::Bicgstab` — for general non-symmetric systems.

Preconditioners:

- `batch::preconditioner::Jacobi` — scalar and block variants.

The vector type is `batch::MultiVector<V>`, *not* `matrix::Dense`. These are distinct types — do
not mix them.

## Construction example

```cpp
namespace b = gko::batch;
auto exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create());

// Build N independent matrices that share sparsity pattern.
std::vector<gko::matrix_data<double, int>> per_item_data = ...;

// Construct the batched matrix.
auto mat = b::matrix::Csr<double, int>::create(exec);
mat->read(per_item_data);   // host data, copies to device

// Build a right-hand side and a solution multivector.
auto rhs = b::MultiVector<double>::create(exec, gko::batch_dim<2>{N, gko::dim<2>{n, 1}});
auto sol = b::MultiVector<double>::create(exec, gko::batch_dim<2>{N, gko::dim<2>{n, 1}});

// Configure and apply a batched CG.
auto factory = b::solver::Cg<double>::build()
                   .with_max_iterations(100)
                   .with_tolerance(1e-8)
                   .with_tolerance_type(b::stop::tolerance_type::absolute)
                   .on(exec);

auto solver = factory->generate(mat);
solver->apply(rhs, sol);
```

After `apply`, `sol` contains all N solutions; per-item iteration counts and residuals are
accessible via `b::log::BatchConvergence`.

## What batched is not

- Not a replacement for distributed solves. The batch dimension is **independent** items, not
  coupled subdomains.
- Not for batches of *different* matrices — the uniform-batch constraint rules that out.
- Not currently for triangular factorisations or AMG (not implemented for batched).

## Batched sub-pages

```{toctree}
:maxdepth: 1
:caption: Batched

batched/per-solver-design
```

## Publications

- [Batched Sparse Iterative Solvers for Computational Chemistry Simulations on GPUs](https://doi.org/10/gn3xcg)
- [Preconditioners for Batched Iterative Linear Solvers on GPUs](https://doi.org/10.1007/978-3-031-23606-8_3)
- [Batched Sparse Iterative Solvers on GPU for the Collision Operator for Fusion Plasma Simulations](https://doi.org/10.1109/IPDPS53621.2022.00024)
- [Integrating Batched Sparse Iterative Solvers for the Collision Operator in Fusion Plasma Simulations on GPUs](https://doi.org/10.1016/j.jpdc.2023.03.012)
- [Accelerating Fusion Plasma Collision Operator Solves with Portable Batched Iterative Solvers on GPUs](https://doi.org/10.1007/978-3-031-73716-9_9)
- [Porting Batched Iterative Solvers onto Intel GPUs with SYCL](https://doi.org/10.1145/3624062.3624181)
- [Efficient Solution of Batched Band Linear Systems on GPUs](https://doi.org/10.1177/10943420251347460)
- [Utilizing Batched Solver Ideas for Efficient Solution of Non-Batched Linear Systems](https://doi.org/10.1109/IPDPSW59300.2023.00113)
- [GPU-based LU Factorization and Solve on Batches of Matrices with Band Structure](https://doi.org/10.1145/3624062.3624247)
- [Evaluating the Performance of NVIDIA's A100 Ampere GPU for Sparse and Batched Computations](https://doi.org/10.1109/PMBS51919.2020.00009)

:::{seealso}
- [Solvers — taxonomy](solvers.md) — single-system solvers; batched lives in parallel.
- [Stopping criteria](stopping-criteria.md) — batched uses simpler compile-time criteria.
- API reference: [`gko::batch`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/namespacegko_1_1batch.html)
:::
