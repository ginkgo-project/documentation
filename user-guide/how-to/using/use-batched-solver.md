# Use a batched solver

Ginkgo's batched solvers solve many small independent linear systems
in a single fused kernel, with no inter-batch communication. The price
is the **uniform-batch constraint**: every system must have the same
dimensions and the same sparsity pattern. When that holds, batched
solvers significantly outperform a loop over single-system solves.

## The recipe

```cpp
#include <ginkgo/ginkgo.hpp>

namespace b = gko::batch;

const gko::size_type N = 100;     // number of batch items
const int n = 256, nnz = 1024;    // rows and non-zeros per item

auto exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create());

// Build the batched matrix from three arrays:
//    - values:   N * nnz entries, item k at offset k * nnz
//    - col_idxs: nnz entries, shared by every item
//    - row_ptrs: n + 1 entries, shared by every item
gko::array<double> values  {exec, N * nnz};
gko::array<int>    col_idxs{exec, nnz};
gko::array<int>    row_ptrs{exec, n + 1};
// ... fill from your application data ...

auto A = gko::share(b::matrix::Csr<double, int>::create(
    exec,
    gko::batch_dim<2>{N, gko::dim<2>{n, n}},
    std::move(values), std::move(col_idxs), std::move(row_ptrs)));

// Build the right-hand side and solution multi-vectors.
gko::batch_dim<2> vec_dim{N, gko::dim<2>{n, 1}};
auto rhs = b::MultiVector<double>::create(exec, vec_dim);
auto sol = b::MultiVector<double>::create(exec, vec_dim);
// ... fill rhs (per-item values laid out the same way as A's values) ...

// Configure and apply a batched CG.
auto solver = b::solver::Cg<double>::build()
    .with_max_iterations(100)
    .with_tolerance(1e-8)
    .with_tolerance_type(b::stop::tolerance_type::absolute)
    .on(exec)
    ->generate(A);

solver->apply(rhs, sol);
```

After `apply`, `sol` holds all `N` solutions; each item lives at
offset `k * n * 1` in the multi-vector's value buffer.

## Pick the solver

| Class | When to use |
|---|---|
| `batch::solver::Cg<V>` | Each item is SPD. |
| `batch::solver::Bicgstab<V>` | Each item is general non-symmetric. |

Both accept a preconditioner via `with_preconditioner`; currently
`batch::preconditioner::Jacobi` is the only batched preconditioner.

## Read per-item convergence

`batch::log::BatchConvergence` captures the final iteration count and
residual norm per item:

```cpp
auto conv = gko::share(b::log::BatchConvergence<double>::create());
solver->add_logger(conv);
solver->apply(rhs, sol);

// Copy to host for inspection. get_num_iterations() returns gko::array<int>
// on the solver's executor.
auto iters_host = gko::array<int>{exec->get_master(),
                                  conv->get_num_iterations()};
auto resn_host  = gko::array<double>{exec->get_master(),
                                     conv->get_residual_norm()};

for (gko::size_type k = 0; k < N; ++k) {
    std::cout << "item " << k
              << " iters=" << iters_host.get_const_data()[k]
              << " ||r||=" << resn_host.get_const_data()[k] << '\n';
}
```

The arrays are sized on first completion and reused on subsequent
applies of the same batch size.

## Wrap application-owned memory

For zero-copy from a buffer your application already owns, use
`create_const`:

```cpp
auto A = b::matrix::Csr<double, int>::create_const(
    exec, mat_size,
    gko::make_const_array_view(exec, N * nnz, values_ptr),
    gko::make_const_array_view(exec, nnz,     col_idxs_ptr),
    gko::make_const_array_view(exec, n + 1,   row_ptrs_ptr));
```

The application is responsible for keeping the buffers alive while
the batched matrix is in use.

## Common pitfalls

- **Uniform-batch constraint.** Items with different sparsity
  patterns or different `nnz` per item are not supported. If your
  patterns differ slightly, pad each item to the union of patterns.
- **Dense input is *not* `gko::matrix::Dense`.** Batched RHS / solution
  vectors are `batch::MultiVector<V>`, a distinct type. Mixing the
  two does not compile.
- **No per-iteration logging.** The batched path runs entirely
  inside the kernel; per-iteration host callbacks would defeat the
  fused design. Only completion is logged.
- **CPU backends.** The fused-kernel design assumes a GPU target.
  Batched solvers compile for OpenMP and Reference but the speedup
  over single-system solves is much smaller there.

:::{seealso}
- [Batched solvers: overview](../../concepts/batched.md) (uniform-batch constraint and storage layout).
- [`batch::solver::Cg`](../../concepts/batched/cg.md), [`batch::solver::Bicgstab`](../../concepts/batched/bicgstab.md), [`batch::log::BatchConvergence`](../../concepts/batched/convergence.md).
:::
