# Add a backend extension

The `extensions/` tree (and the corresponding public headers under
`include/ginkgo/extensions/`) holds header-only adapters that let Ginkgo
interoperate with external frameworks. The shipped extensions are:

| Extension | Purpose | Public header |
|---|---|---|
| Kokkos | Map `gko::array<T>` and `matrix::Dense<T>` to `Kokkos::View`; align Ginkgo executors with Kokkos execution / memory spaces | `ginkgo/extensions/kokkos.hpp` |
| nlohmann JSON config | Parse a `nlohmann::json` tree into a `gko::config::pnode` for the runtime factory registry | `ginkgo/extensions/config/json_config.hpp` |
| yaml-cpp YAML config | Same, for YAML | `ginkgo/extensions/config/yaml_config.hpp` |

Extensions are not "backends" in the kernel-dispatch sense (those live
under `cuda/`, `hip/`, `omp/`, `dpcpp/`). They are user-facing libraries
that depend on third-party packages and provide bidirectional bridges
between Ginkgo and that package.

## What an extension provides

The shipped extensions all share the same shape:

- **A single public umbrella header** (`ginkgo/extensions/<name>.hpp`)
  that pulls in the sub-headers.
- **Sub-headers** under `ginkgo/extensions/<name>/` for each concern
  (Kokkos splits into `spaces.hpp`, `types.hpp`, `config.hpp`).
- **A namespace**: `gko::ext::<name>`. The bridge functions and types
  all live there.
- **No `.cpp` file**: the extension is header-only and depends on
  Ginkgo's public API plus the third-party package. The third-party
  package is found by the consumer at configure time, not by Ginkgo's
  build.
- **Optional `config.hpp.in`** processed by `configure_file` so the
  extension can see Ginkgo's compile-time switches (e.g. whether the
  CUDA backend was built).

## Anatomy of the Kokkos extension

The umbrella header is the entry point users include:

```cpp
// include/ginkgo/extensions/kokkos.hpp
#include <ginkgo/extensions/kokkos/config.hpp>
#include <ginkgo/extensions/kokkos/spaces.hpp>
#include <ginkgo/extensions/kokkos/types.hpp>
```

The two substantive sub-headers do the bridging:

- `kokkos/spaces.hpp` defines `compatible_space<MemorySpace, ExecType>`
  — a `std::integral_constant<bool, ...>` that answers "can this
  Kokkos memory space be safely accessed from this Ginkgo executor?".
  This is what `map_executor` uses internally to refuse mismatched
  pairs at compile time.
- `kokkos/types.hpp` defines `value_type<T>::type` — `Kokkos::complex<T>`
  for `std::complex<T>`, identity for everything else — and a
  `View`-returning helper that wraps a `gko::array<T>` or
  `matrix::Dense<T>` storage region as a non-owning Kokkos `View`.

Both files live under `namespace gko::ext::kokkos`, with internal
implementation details in `namespace gko::ext::kokkos::detail`.

The `config.hpp.in` is processed by Ginkgo's top-level CMakeLists:

```cmake
# CMakeLists.txt
configure_file(
    ${Ginkgo_SOURCE_DIR}/include/ginkgo/extensions/kokkos/config.hpp.in
    ${Ginkgo_BINARY_DIR}/include/ginkgo/extensions/kokkos/config.hpp
    @ONLY
)
```

It substitutes Ginkgo's build switches (e.g. `GINKGO_BUILD_CUDA`,
`GINKGO_BUILD_OMP`) into the generated `config.hpp` so the Kokkos
spaces header can guard CUDA-specific bridges behind the right macros.

## How extensions plug into the build

The top-level Ginkgo CMakeLists has a single `add_subdirectory(extensions)`
line (after the core build is in place). The `extensions/CMakeLists.txt`
itself is one block:

```cmake
if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()
```

There is **no library target for the extension itself** — it's
header-only. The only thing the build system needs to do is build the
tests, and that happens conditionally on `GINKGO_BUILD_TESTS`.

The per-extension test directories are where the third-party dependency
is actually fetched. The JSON / YAML config test directory uses
`FetchContent` to pull `yaml-cpp` if `find_package(yaml-cpp)` fails;
the Kokkos test directory uses `find_package(Kokkos)` and asserts that
the backends Kokkos was built with match the Ginkgo backends in use.

## Adding a new extension

The pattern for a new extension `<myext>` is fixed. You add:

```
include/ginkgo/extensions/<myext>.hpp              ← umbrella include
include/ginkgo/extensions/<myext>/types.hpp        ← per-concern headers
include/ginkgo/extensions/<myext>/spaces.hpp
include/ginkgo/extensions/<myext>/config.hpp.in    ← optional, if you need
                                                     Ginkgo build-flag knowledge
extensions/test/<myext>/CMakeLists.txt             ← finds your third-party dep
extensions/test/<myext>/<test1>.cpp                ← tests live here
```

CMake wiring:

- If `<myext>` needs a generated `config.hpp`, add a `configure_file`
  block to the top-level `CMakeLists.txt` next to the Kokkos one.
- Append your test subdirectory to `extensions/test/CMakeLists.txt`.
- Inside `extensions/test/<myext>/CMakeLists.txt`, call
  `find_package(<MyExtDep> REQUIRED)` (with a `FetchContent` fallback if
  appropriate), define a `ginkgo_create_test` target for each test
  file, and link `Ginkgo::ginkgo` plus the third-party target.

Because extensions are header-only, there is no `.cpp` file to register
with the core build's source list. Users opting in to the extension
include the corresponding header and call `find_package(Ginkgo)` plus
`find_package(<MyExtDep>)` in their own CMake.

## Wrapping a vendor library as a LinOp — where it lives today

A different pattern that often comes up in the same context is wrapping
a vendor library (cuSPARSE, cuSolverSp, MAGMA, ...) as a `LinOp` so it
participates in Ginkgo's solver and preconditioner composition. The
shipped examples — `solver::LowerTrs` / `solver::UpperTrs` with their
cuSPARSE triangular-solve backend, `factorization::Cholesky` with its
optional vendor path — live as ordinary core types in `core/solver/`
and `core/factorization/`, with the vendor calls confined to backend
kernel files (`cuda/solver/lower_trs_kernels.cu`,
`cuda/factorization/cholesky_kernels.cu`).

In other words: today, the `extensions/` tree is for header-only
framework interop. A vendor-library-backed `LinOp` is a regular core
type, written the same way as any other solver (see [Add a new
solver](add-a-solver.md)) or preconditioner (see [Add a new
preconditioner](add-a-preconditioner.md)). The kernel files in the
relevant backend (`cuda/`, `hip/`, `dpcpp/`) include the vendor headers
and own the vendor handle as a private member, opaquely from the
public class.

:::{seealso}
- [Add a new solver](add-a-solver.md) — for the LinOp + Factory + parse
  scaffolding that any vendor-backed solver also needs.
- [Add a new preconditioner](add-a-preconditioner.md) — same scaffolding,
  no workspace.
- [Configure via JSON](../using/configure-via-json.md) — what the JSON
  config extension unlocks for the runtime registry.
:::
