# cuDSS direct solver

`gko::ext::cuda::solver::Cudss` is a wrapper around NVIDIA's cuDSS sparse direct
solver. It is only available on the `CudaExecutor` and only when CMake found cuDSS 0.8.0 or newer
while configuring Ginkgo (see [Build-time enablement](#build-time-enablement) below).
It is the recommended path when you need a fast sparse direct solver on NVIDIA GPUs
and Ginkgo's own LU/Cholesky factorization does not perform well for your matrix.

## When to use it

cuDSS is appropriate when:

- You are running on NVIDIA hardware and want vendor-tuned sparse factorization /
  triangular solve performance.
- Your matrix has a structure that benefits from cuDSS's reordering and pivoting
  heuristics.
- You need to **refactorize** repeatedly with the same sparsity pattern but different
  numerical values (e.g., a Newton iteration that updates Jacobian values without
  reshaping the system) — cuDSS reuses both the symbolic analysis and the
  factorization data structures across `refactorize` calls, updating only the
  numerical values in place. Ginkgo's own `experimental::solver::Direct` can skip
  the symbolic step (by passing a previously computed `symbolic_factorization` to
  the `Lu` / `Cholesky` factory it wraps), but the numeric factorization and the
  solver itself are rebuilt each time, so the saving is smaller than cuDSS's
  in-place refactorization.

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

## Factory parameters

| Parameter | Type | Purpose |
|---|---|---|
| `matrix_type` | `int` | The mathematical structure of the matrix (`cudssMatrixType_t`): `0` GENERAL (default), `1` SYMMETRIC, `2` HERMITIAN, `3` SPD, `4` HPD. |
| `matrix_view` | `int` | What the CSR storage actually contains (`cudssMatrixViewType_t`): `0` FULL (default), `1` LOWER, `2` UPPER. |
| `reordering_alg` | `int` | Fill-reducing reordering, passed to cuDSS as `CUDSS_CONFIG_REORDERING_ALG` (`cudssReorderingAlg_t`). `0` (default) selects cuDSS's default algorithm. |
| `hybrid_execute` | `bool` | Enables cuDSS hybrid host/device execution (`CUDSS_CONFIG_HYBRID_EXECUTE_MODE`). Defaults to `false`. |
| `hybrid_memory` | `bool` | Enables cuDSS hybrid memory mode, keeping parts of the factorization in host memory (`CUDSS_CONFIG_HYBRID_MEMORY_MODE`). Defaults to `false`. |

All cuDSS calls are issued on the `CudaExecutor`'s stream.

:::{attention}
Storage-format mismatch between Ginkgo and cuDSS:

- `gko::matrix::Csr` stores the **full** matrix.
- cuDSS's symmetric / Hermitian / SPD / HPD modes assume **only one
  triangle** is stored.
- Passing a fully-stored symmetric matrix with `matrix_type` set to a
  symmetric mode violates cuDSS's input contract and can produce
  wrong results.

Two correct paths:

- **Fully-stored matrix** — `matrix_type = 0` (GENERAL),
  `matrix_view = 0` (FULL).
- **Symmetric factorisation** — extract one triangle into a new CSR
  first, then construct the solver with the symmetric mode.
:::

## Refactorization with the same sparsity pattern

When the matrix's sparsity pattern is fixed but its numerical values change between
solves, calling `refactorize(new_matrix)` updates the numeric factorization without
re-running the symbolic analysis:

```cpp
#include <ginkgo/extensions/cuda/solver/cudss.hpp>

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

The cuDSS extension is built automatically
when both of the following hold at configure time:

1. Ginkgo is built with CUDA support (`-DGINKGO_BUILD_CUDA=ON`).
2. CMake's `find_package(cudss 0.8.0 CONFIG)` discovers cuDSS 0.8.0 or newer.

If cuDSS is not in a default search location, point CMake at it explicitly:

```bash
cmake -DGINKGO_BUILD_CUDA=ON -Dcudss_DIR=/path/to/cudss/lib/cmake/cudss ...
```

The solver lives in its own library. Link it in addition to Ginkgo:

```cmake
find_package(Ginkgo REQUIRED)
target_link_libraries(app PRIVATE Ginkgo::ginkgo Ginkgo::ginkgo_cudss)
```

If cuDSS is not found, the build emits a `STATUS` message and silently skips the
extension — `gko::ext::cuda::solver::Cudss` will not be available, but the rest of
Ginkgo continues to build normally.

## Configuration files

`Cudss` can be created from a [configuration file](../config-files.md) once its entry is added to the registry:

```cpp
#include <ginkgo/extensions/cuda/solver/cudss.hpp>

using Cudss = gko::ext::cuda::solver::Cudss<double, gko::int32>;
gko::config::registry reg{Cudss::get_config_map()};
auto factory = gko::config::parse(config, reg).on(exec);
```

```json
{"type": "ext::cuda::solver::Cudss", "matrix_type": 0, "matrix_view": 0,
 "reordering_alg": 0, "hybrid_execute": false, "hybrid_memory": false}
```

Keys other than `type` and the factory parameters above raise an error; the entry creates the `Cudss<ValueType, IndexType>` whose `get_config_map()` you registered.

## Limitations

- **CUDA only** — the extension targets NVIDIA GPUs through the `CudaExecutor`.
- **Opaque factorization** — the L and U factors are stored in cuDSS-native format and
  cannot be extracted as Ginkgo `LinOp`s.
- **Build-time discovery** — requires CUDA support plus a CMake-discoverable cuDSS
  0.8.0+ installation.

:::{seealso}
- [Direct (LU / Cholesky)](../solvers/direct.md) — the cross-backend Ginkgo direct solver.
- [Solvers — taxonomy](../solvers.md) — where direct solvers fit in the broader picture.
- [Reordering and permutations](../reordering.md) — apply a fill-reducing reordering before factorization.
- API reference: [`gko::ext::cuda::solver::Cudss`](../../../api-reference/extensions/cudss.md)
:::
