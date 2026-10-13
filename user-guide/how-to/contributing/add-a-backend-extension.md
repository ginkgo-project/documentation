# Add a backend extension

The `extensions/` tree (and the corresponding public headers under
`include/ginkgo/extensions/`) holds user-facing libraries that depend
on third-party packages and bridge them with Ginkgo. They are not
"backends" in the kernel-dispatch sense — those live under `cuda/`,
`hip/`, `omp/`, `dpcpp/`.

Shipped extensions:

| Extension | Shape | Public header |
|---|---|---|
| Kokkos | Header-only | `ginkgo/extensions/kokkos.hpp` |
| nlohmann JSON config | Header-only | `ginkgo/extensions/config/json_config.hpp` |
| yaml-cpp YAML config | Header-only | `ginkgo/extensions/config/yaml_config.hpp` |
| cuDSS direct solver | Compiled library (`ginkgo_cudss`) | `ginkgo/extensions/cuda/solver/cudss.hpp` |

An extension can be **header-only** (Kokkos / JSON / YAML — pure
bridging headers that consumers `#include`) or a **compiled library
target** (cuDSS — wraps a vendor library as a `LinOp`, distributed as
its own `Ginkgo::<name>` library alongside `Ginkgo::ginkgo`). The
right shape depends on what the extension does.

## What an extension provides

Every extension, regardless of shape:

- Lives under `namespace gko::ext::<name>` (e.g.
  `gko::ext::kokkos`, `gko::ext::cuda::solver`). Internal details go
  in a `detail` sub-namespace.
- Has a public header tree under `include/ginkgo/extensions/<name>/`.
  Header-only extensions usually expose a single umbrella header
  `extensions/<name>.hpp` that pulls in the sub-headers.
- Optionally has a `config.hpp.in` template processed by
  `configure_file` so the extension can see Ginkgo's compile-time
  switches.

Header-only extensions stop there. Compiled-library extensions
additionally have:

- One or more `.cpp` files under `extensions/<area>/` (e.g.
  `extensions/cuda/solver/cudss.cpp`).
- A library target produced by `add_library(...)` in the relevant
  `extensions/<area>/CMakeLists.txt`, conventionally aliased as
  `Ginkgo::<name>`.

## Anatomy of the Kokkos extension (header-only)

Umbrella header:

```cpp
// include/ginkgo/extensions/kokkos.hpp
#include <ginkgo/extensions/kokkos/config.hpp>
#include <ginkgo/extensions/kokkos/spaces.hpp>
#include <ginkgo/extensions/kokkos/types.hpp>
```

The two substantive sub-headers do the bridging:

- `kokkos/spaces.hpp` — defines `detail::compatible_space<MemorySpace,
  ExecType>`, a `std::integral_constant<bool, ...>` that answers "can
  this Kokkos memory space be safely accessed from this Ginkgo
  executor?". `map_data` checks it through `assert_compatibility`,
  which throws at run time (`GKO_THROW_IF_INVALID`) for a mismatched
  pair. The header also provides `create_executor`,
  `create_default_executor` and `create_default_host_executor` to
  build Ginkgo executors from Kokkos execution spaces.
- `kokkos/types.hpp` — defines `detail::value_type<T>::type`
  (`Kokkos::complex<T>` for `std::complex<T>`, identity otherwise)
  plus `map_data`, which wraps a `gko::array<T>` or `matrix::Dense<T>`
  storage region as a non-owning Kokkos `View`.

The `config.hpp.in` is processed by Ginkgo's top-level CMakeLists:

```cmake
configure_file(
    ${Ginkgo_SOURCE_DIR}/include/ginkgo/extensions/kokkos/config.hpp.in
    ${Ginkgo_BINARY_DIR}/include/ginkgo/extensions/kokkos/config.hpp
    @ONLY
)
```

It substitutes the `GINKGO_EXTENSION_KOKKOS_CHECK_TYPE_ALIGNMENT` option
into the generated `config.hpp` so the Kokkos headers can switch their
type-alignment checks on or off.

## Anatomy of the cuDSS extension (compiled library)

cuDSS wraps NVIDIA's sparse direct solver as
`gko::ext::cuda::solver::Cudss<V, I>` — an ordinary `LinOp` with the
standard factory/parse scaffolding. Layout:

```text
extensions/cuda/CMakeLists.txt                 ← find_package + add_library
extensions/cuda/solver/cudss.cpp               ← implementation
include/ginkgo/extensions/cuda/solver/cudss.hpp ← public LinOp class
```

The CMake target is built only when the prerequisites are present:

```cmake
# extensions/cuda/CMakeLists.txt
if(NOT GINKGO_BUILD_CUDA)
    return()
endif()
if(NOT cudss_FOUND)
    message(
        STATUS
        "cuDSS 0.8.0 or newer not found — gko::ext::cuda::solver::Cudss will not be built"
    )
    return()
endif()

add_library(ginkgo_cudss solver/cudss.cpp)
add_library(Ginkgo::ginkgo_cudss ALIAS ginkgo_cudss)

target_link_libraries(ginkgo_cudss PUBLIC ginkgo)
target_link_libraries(ginkgo_cudss PRIVATE cudss CUDA::cudart)

ginkgo_default_includes(ginkgo_cudss)
ginkgo_install_library(ginkgo_cudss)
```

The vendor dependency is discovered at the `extensions/` top level:

```cmake
# extensions/CMakeLists.txt
if(GINKGO_BUILD_CUDA)
    find_package(cudss 0.8.0 CONFIG)
endif()
add_subdirectory(cuda)
```

Notes on this pattern:

- The library links **publicly** to `ginkgo` (so consumers get the
  public Ginkgo headers transitively) and **privately** to the vendor
  package and CUDA runtime (so consumers don't have to discover them).
- The `Ginkgo::ginkgo_cudss` alias is what users link against.
- `ginkgo_default_includes` + `ginkgo_install_library` give the
  extension the same install layout and include-path setup as the
  core Ginkgo target.

The wrapped class itself is a regular `LinOp`: `Cudss` derives from
`LinOp` and has the standard `parameters_type` / `Factory`
/ `parse` triplet (see [Add a new solver](add-a-solver.md) for the
scaffolding). The cuDSS handles and factorization state are private
members, opaque to the public class.

## How extensions plug into the build

The top-level Ginkgo `CMakeLists.txt` calls
`add_subdirectory(extensions)` after the core build. The
`extensions/CMakeLists.txt` is small: it does the vendor-package
discovery shared across sub-extensions, walks into the
compiled-library subdirectories (e.g. `cuda/`), and conditionally
walks into `test/` when `GINKGO_BUILD_TESTS=ON`:

```cmake
# extensions/CMakeLists.txt
if(GINKGO_BUILD_CUDA)
    find_package(cudss 0.8.0 CONFIG)
endif()

add_subdirectory(cuda)
if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()
```

What gets built depends on the extension:

- **Header-only extensions** contribute nothing to the build —
  consumers just `#include` the header. Their tests live under
  `extensions/test/<name>/`. `extensions/test/CMakeLists.txt` guards the
  Kokkos tests with `find_package(Kokkos QUIET)`. The config tests
  always build: `extensions/test/config/CMakeLists.txt` finds yaml-cpp
  with `find_package(... QUIET)` and falls back to `FetchContent`,
  nlohmann_json falls back to `third_party/`, and
  `ginkgo_create_test(... ADDITIONAL_LIBRARIES ...)` links them.
- **Compiled-library extensions** add a library target under
  `extensions/<area>/CMakeLists.txt`, conditional on the vendor
  package being found. The build skips the target when the
  vendor dependency is missing (cuDSS prints an informational
  `STATUS` message and returns).

## Adding a new extension

For a header-only extension `<myext>`:

```text
include/ginkgo/extensions/<myext>.hpp              ← umbrella include
include/ginkgo/extensions/<myext>/types.hpp        ← per-concern headers
include/ginkgo/extensions/<myext>/spaces.hpp
include/ginkgo/extensions/<myext>/config.hpp.in    ← optional, if you need
                                                     Ginkgo build-flag knowledge
extensions/test/<myext>/CMakeLists.txt             ← links your third-party dep
extensions/test/<myext>/<test1>.cpp                ← tests live here
```

CMake wiring:

- If `<myext>` needs a generated `config.hpp`, add a `configure_file`
  block to the top-level `CMakeLists.txt` next to the Kokkos one.
- In `extensions/test/CMakeLists.txt`, wrap
  `add_subdirectory(<myext>)` in a `find_package(<MyExtDep> QUIET)`
  guard, so the tests are built only when the dependency is found.
- In `extensions/test/<myext>/CMakeLists.txt`, call
  `ginkgo_create_test(<test1> ADDITIONAL_LIBRARIES <MyExtDep::target>)`
  to link the third-party target into each test. Add a
  `find_package(... QUIET)` plus `FetchContent` fallback there when the
  tests should build without a preinstalled dependency.

For a compiled-library extension `<myext>` (vendor `LinOp` wrapper):

```text
include/ginkgo/extensions/<area>/<myext>.hpp       ← public LinOp class
extensions/<area>/<myext>.cpp                      ← implementation
extensions/<area>/CMakeLists.txt                   ← add_library + linking
extensions/test/<area>/CMakeLists.txt              ← ginkgo_create_test
extensions/test/<area>/solver/<myext>.cpp          ← tests
```

CMake wiring:

- `find_package(<MyVendor> ...)` in `extensions/CMakeLists.txt` (so
  every compiled sub-extension under `extensions/<area>/` can see
  whether the dependency is present).
- `add_subdirectory(<area>)` from `extensions/CMakeLists.txt`.
- In `extensions/<area>/CMakeLists.txt`, early-return when the
  required Ginkgo backend isn't enabled and when the vendor package
  isn't found. Then `add_library(ginkgo_<myext> <myext>.cpp)`,
  `add_library(Ginkgo::ginkgo_<myext> ALIAS ginkgo_<myext>)`, public
  link to `ginkgo`, private link to the vendor target, and
  `ginkgo_default_includes` + `ginkgo_install_library`.
- Add `add_subdirectory(<area>)` to `extensions/test/CMakeLists.txt`
  (cuDSS's `cuda` directory is added under `if(GINKGO_BUILD_CUDA)`).
  In `extensions/test/<area>/CMakeLists.txt`, return early when the
  backend or vendor package is missing, then call
  `ginkgo_create_test(solver/<myext> ADDITIONAL_LIBRARIES
  ginkgo_<myext>)`, as `extensions/test/cuda/CMakeLists.txt` does for
  cuDSS.

The class itself is built like any other `LinOp` — see [Add a new
solver](add-a-solver.md) or [Add a new
preconditioner](add-a-preconditioner.md) for the factory / parse
scaffolding.

## Where to place a vendor-LinOp wrapper

Vendor library wrappers can live in `core/` when they belong to a
shipped Ginkgo type, or in `extensions/` when they stand alone:

- **In `core/`** — when the vendor path is one implementation among
  several for a type that Ginkgo already exposes. Example:
  `solver::LowerTrs` / `solver::UpperTrs` expose the choice through
  their `algorithm` parameter: `trisolve_algorithm::sparselib`
  (the default) uses the vendor library, while `syncfree` runs
  Ginkgo's own kernel on CUDA. The vendor calls live in
  `cuda/solver/lower_trs_kernels.cu` etc., gated by the executor
  type.
- **In `extensions/`** — when the type is *only* the vendor wrapper
  and would not exist without it. Example: `gko::ext::cuda::solver::Cudss`
  is implemented only through cuDSS, so it is its own extension library
  that consumers opt into.

:::{seealso}
- [Add a new solver](add-a-solver.md) — LinOp + Factory + parse
  scaffolding that any vendor-backed solver also needs.
- [Add a new preconditioner](add-a-preconditioner.md) — same
  scaffolding, no workspace.
- [Configure via JSON](../using/configure-via-json.md) — what the JSON
  config extension unlocks for the runtime registry.
:::
