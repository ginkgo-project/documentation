# Batched solvers: overview

For workloads that need to solve many small independent linear systems (flame chemistry,
multiphase Lagrangian particles, parameter sweeps), Ginkgo provides a batched subsystem.
All systems are solved simultaneously inside a single fused GPU kernel, with no inter-batch
communication. This page covers when batched is the right tool, the uniform-batch constraint,
the storage layout, and the fused-kernel design.

## When to use batched

Batched solvers are suited to:

- Many systems (`> 100`) of small size (typically `≤ 10⁴` rows each).
- All systems are independent: no coupling between batch items.
- Same matrix dimensions and same sparsity pattern across items (the uniform-batch constraint).

If those conditions hold, batched almost always outperforms looping over single-system solves:
kernel launch overhead is paid once instead of N times, and memory access can be coalesced
across batch items.

## The uniform-batch constraint

:::{attention}
All items in a batch must share:

- the same matrix dimensions (`num_rows`, `num_cols`),
- the same sparsity pattern: col indices and row pointers are stored **once**, shared across the
  batch,
- the same number of non-zeros per item.
:::

This is restrictive but it's what enables the fused-kernel design.

## Storage layout

For batched CSR, two arrays are batch-shared and one is per-item:

- `col_idxs`: single shared copy, length `nnz`.
- `row_ptrs`: single shared copy, length `num_rows + 1`.
- `values`: per-item, total length `batch_items * nnz`. Item `k`'s values start at offset
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

The defining feature of batched solvers: the entire iterative solve, every iteration
included, runs inside a single device kernel launch.

- **One workgroup per batch item.** A *workgroup* is the portable name for what CUDA calls a
  *thread block* and HIP and SYCL also call a *work-group*, a fixed-size cluster of threads
  that share fast on-chip memory and can synchronise cheaply. Each batch item is solved
  independently inside its own workgroup.
- **Sub-group reductions.** Within a workgroup the kernels further partition into a *sub-group*
  (CUDA *warp* / HIP *wavefront* / SYCL *sub-group*) using cooperative-groups
  `tiled_partition<tile_size>(thread_block)`; the sub-group handles row-level dot products and
  norms. The sub-group size is chosen at compile time per-backend.
- **Backend-portable barriers.** The CUDA / HIP kernels synchronise the workgroup with
  `__syncthreads()`; the SYCL kernels use the equivalent `item.barrier(...)`. From the
  high-level kernel source the call is the same; only the backend dispatch layer picks the
  matching primitive.
- The iteration loop runs **inside** the kernel: no per-iteration kernel launch.
- Scalars (rho, alpha, residual norms) live in shared / local memory.
- Vectors (residual, search direction) are split between shared and global memory based on a
  `storage_config` heuristic that balances against available shared memory.

Why this matters: kernel launch overhead disappears, and shared-memory residency means
inner-loop reductions are fast.

## What's available

Matrix formats:

- `batch::matrix::Dense`
- `batch::matrix::Csr`
- `batch::matrix::Ell`
- `batch::matrix::Identity`: for the no-matrix (identity preconditioner) case.

Solvers:

- `batch::solver::Cg`: for symmetric positive-definite systems.
- `batch::solver::Bicgstab`: for general non-symmetric systems.

Preconditioners:

- `batch::preconditioner::Jacobi`: scalar and block variants.

The vector type is `batch::MultiVector<V>`, *not* `matrix::Dense`. These are distinct types; do
not mix them.

## Construction example

A batched matrix is built from three pre-packed `gko::array`s, namely the values concatenated across
all items, plus the single shared `col_idxs` and `row_ptrs` for one item:

```cpp
namespace b = gko::batch;
auto exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create());

// Per-item dimensions and total batch size.
const gko::size_type N = 100;          // number of items in the batch
const int n = 256, nnz = 1024;          // rows and non-zeros per item
gko::batch_dim<2> mat_size{N, gko::dim<2>{n, n}};

// Three contiguous device-resident arrays:
//   values    — N * nnz entries, item k's values at offset k * nnz
//   col_idxs  — nnz entries, shared by every item
//   row_ptrs  — n + 1 entries, shared by every item
gko::array<double> values{exec, N * nnz};
gko::array<int>    col_idxs{exec, nnz};
gko::array<int>    row_ptrs{exec, n + 1};
// ... fill the arrays from your application data ...

auto mat = gko::share(b::matrix::Csr<double, int>::create(
    exec, mat_size, std::move(values), std::move(col_idxs), std::move(row_ptrs)));

// Build a right-hand side and a solution multivector.
gko::batch_dim<2> vec_size{N, gko::dim<2>{n, 1}};
auto rhs = b::MultiVector<double>::create(exec, vec_size);
auto sol = b::MultiVector<double>::create(exec, vec_size);
// ... fill rhs / initial guess ...

// Configure and apply a batched CG.
auto solver = b::solver::Cg<double>::build()
                  .with_max_iterations(100)
                  .with_tolerance(1e-8)
                  .with_tolerance_type(b::stop::tolerance_type::absolute)
                  .on(exec)
                  ->generate(mat);
solver->apply(rhs, sol);
```

The shared `col_idxs` and `row_ptrs` are the structural promise of the
[uniform-batch constraint](#the-uniform-batch-constraint): every item is required to have the
same sparsity pattern, so storing the pattern once is correct by construction.

If your application has the systems as `std::vector<gko::matrix_data<V, I>>` instead of
pre-packed arrays, build a single-item `gko::matrix::Csr` from each `matrix_data` via its
`read(matrix_data)` method, then concatenate the per-item values into a contiguous buffer; the
shared `col_idxs` and `row_ptrs` come from any one of the per-item CSRs since they are required
to match. For zero-copy integration with application-owned memory, use `create_const(...)`
instead.

After `apply`, `sol` contains all N solutions; per-item iteration counts and residuals are
accessible by attaching a [`b::log::BatchConvergence`](batched/convergence.md) logger before
the apply call.

## What batched is not

- Not a replacement for distributed solves. The batch dimension is **independent** items, not
  coupled subdomains.
- Not for batches of *different* matrices; the uniform-batch constraint rules that out.
- Not currently for triangular factorisations or AMG (not implemented for batched).

## Batched sub-pages

```{toctree}
:maxdepth: 1
:caption: Batched

batched/cg
batched/bicgstab
batched/convergence
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

:::{seealso}
- [Solver taxonomy](solvers.md): single-system solvers; batched lives in parallel.
- [Stopping criteria](stopping-criteria.md): batched uses simpler compile-time criteria.
- API reference: [`gko::batch`](../../api-reference/batch/index.md)
:::
