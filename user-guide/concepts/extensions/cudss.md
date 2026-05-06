# cuDSS direct solver

`gko::ext::cuda::solver::Cudss` is a wrapper around NVIDIA's cuDSS sparse direct
solver. It is only available on the `CudaExecutor` and only when CMake found a cuDSS
installation while configuring Ginkgo (see [Build-time enablement](#build-time-enablement) below).
It is the recommended path when you need a fast sparse direct solver on NVIDIA GPUs
and Ginkgo's own LU/Cholesky factorization is not sufficient — for example, if you
need cuDSS-specific features like in-place refactorization.

## When to use it

cuDSS is appropriate when:

- You are running on NVIDIA hardware and want vendor-tuned sparse factorization /
  triangular solve performance.
- Your matrix has a structure that benefits from cuDSS's reordering and pivoting
  heuristics.
- You need to **refactorize** repeatedly with the same sparsity pattern but different
  numerical values (e.g., a Newton iteration that updates Jacobian values without
  reshaping the system) — cuDSS reuses the symbolic analysis across `refactorize`
  calls, while Ginkgo's own `experimental::solver::Direct` would need to rebuild the
  factorization from scratch.

For non-CUDA backends, fall back to [`gko::experimental::solver::Direct`](../solvers/direct.md).

## Construction

```cpp
#include <ginkgo/extensions/cuda/solver/cudss.hpp>

auto exec = gko::CudaExecutor::create(0, gko::ReferenceExecutor::create());

auto solver = gko::ext::cuda::solver::Cudss<double, gko::int32>::build()
                  .with_matrix_type(0)   // 0 = GENERAL (unsymmetric)
                  .with_matrix_view(0)   // 0 = FULL (entire matrix stored)
                  .on(exec)
                  ->generate(system_matrix);

solver->apply(b, x);
```

The factorization is computed during `generate()` and reused across `apply` calls.

## Matrix type and view

cuDSS exposes two parameters that control how it interprets the supplied CSR matrix:

| Parameter | Values | Meaning |
|---|---|---|
| `matrix_type` | `0` GENERAL, `1` SYMMETRIC, `2` HERMITIAN, `3` SPD, `4` HPD | The mathematical structure of the matrix. |
| `matrix_view` | `0` FULL, `1` LOWER, `2` UPPER | What the CSR storage actually contains: full matrix, only lower triangle + diagonal, or only upper triangle + diagonal. |

:::{attention}
Ginkgo's `gko::matrix::Csr` stores the **full** matrix. cuDSS's symmetric / Hermitian /
SPD / HPD modes assume only one triangle is stored. Passing a fully-stored symmetric
matrix with `matrix_type` set to one of these symmetric modes is **not** the
documented input contract and can produce incorrect results. To use the symmetric
factorizations, extract the upper or lower triangle into a new CSR before
constructing the solver. For a fully-stored matrix, use `matrix_type = 0` (GENERAL)
together with `matrix_view = 0` (FULL).
:::

## Refactorization with the same sparsity pattern

When the matrix's sparsity pattern is fixed but its numerical values change between
solves, calling `refactorize(new_matrix)` updates the numeric factorization without
re-running the symbolic analysis:

```cpp
solver->refactorize(updated_matrix);   // same sparsity, new values
solver->apply(b, x);
```

The new matrix must have the same dimensions and number of non-zeros as the matrix
used in `generate()`. This is much cheaper than rebuilding the solver from scratch.

## Supported value and index types

| Trait | Supported types |
|---|---|
| `ValueType` | `float`, `double`, `std::complex<float>`, `std::complex<double>` |
| `IndexType` | `gko::int32` (default), `gko::int64` (subject to cuDSS support) |

## Build-time enablement

The cuDSS extension does not have a dedicated CMake flag. It is built automatically
when both of the following hold at configure time:

1. Ginkgo is built with CUDA support (`-DGINKGO_BUILD_CUDA=ON`).
2. CMake's `find_package(cudss 0.7.1 CONFIG)` discovers a cuDSS installation.

If cuDSS is not in a default search location, point CMake at it explicitly:

```bash
cmake -DGINKGO_BUILD_CUDA=ON -Dcudss_DIR=/path/to/cudss/lib/cmake/cudss ...
```

If cuDSS is not found, the build emits a `STATUS` message and silently skips the
extension — `gko::ext::cuda::solver::Cudss` will not be available, but the rest of
Ginkgo continues to build normally.

## Limitations

- **CUDA only** — there is no AMD or Intel equivalent in the extensions tree.
- **Opaque factorization** — the L and U factors are stored in cuDSS-native format and
  cannot be extracted as Ginkgo `LinOp`s.
- **Build-time discovery** — requires CUDA support plus a CMake-discoverable cuDSS
  installation; no `GINKGO_BUILD_EXT_CUDSS` flag exists.

:::{seealso}
- [Direct (LU / Cholesky)](../solvers/direct.md) — the cross-backend Ginkgo direct solver.
- [Solvers — taxonomy](../solvers.md) — where direct solvers fit in the broader picture.
- [Reordering and permutations](../reordering.md) — apply a fill-reducing reordering before factorization.
- API reference: [`gko::ext::cuda::solver::Cudss`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1ext_1_1cuda_1_1solver_1_1Cudss.html)
:::
