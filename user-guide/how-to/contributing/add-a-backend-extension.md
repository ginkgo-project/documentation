# Add a backend extension

The `include/ginkgo/extensions/` tree (with its tests under
`extensions/`) holds user-facing headers that depend on third-party
packages and bridge them with Ginkgo. They are not "backends" in the
kernel-dispatch sense — those live under `cuda/`, `hip/`, `omp/`,
`dpcpp/`.

Shipped extensions:

| Extension | Shape | Public header |
|---|---|---|
| Kokkos | Header-only | `ginkgo/extensions/kokkos.hpp` |
| nlohmann JSON config | Header-only | `ginkgo/extensions/config/json_config.hpp` |
| yaml-cpp YAML config | Header-only | `ginkgo/extensions/config/yaml_config.hpp` |

The extensions shipped with Ginkgo are **header-only**: pure bridging
headers that consumers `#include`. Ginkgo installs them with the other
public headers, and the consumer's build links the third-party package.

## What an extension provides

Every extension:

- Lives under `namespace gko::ext::<name>` (e.g. `gko::ext::kokkos`,
  `gko::ext::config`). Internal details go in a `detail`
  sub-namespace.
- Has a public header tree under `include/ginkgo/extensions/<name>/`
  and, for larger extensions, an umbrella header
  `extensions/<name>.hpp` that pulls in the sub-headers.
- Optionally has a `config.hpp.in` template processed by
  `configure_file` so the extension can see Ginkgo's compile-time
  switches.
- Has its tests under `extensions/test/<name>/`. The Kokkos tests are
  built when Kokkos is found; the config tests are built whenever tests
  are enabled, because yaml-cpp and nlohmann_json are fetched when
  missing.

## Anatomy of the Kokkos extension

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
  executor?". `map_data` checks it through `assert_compatibility`, which
  throws at run time (`GKO_THROW_IF_INVALID`) for a mismatched pair. The header also provides
  `create_executor`, `create_default_executor` and
  `create_default_host_executor`, which build a Ginkgo executor
  matching a Kokkos execution space.
- `kokkos/types.hpp` — defines `detail::value_type<T>::type`
  (`Kokkos::complex<T>` for `std::complex<T>`, identity otherwise)
  and `map_data`, which wraps a `gko::array<T>` or `matrix::Dense<T>`
  storage region as a non-owning Kokkos `View`.

The `config.hpp.in` is processed by Ginkgo's top-level CMakeLists:

```cmake
configure_file(
    ${Ginkgo_SOURCE_DIR}/include/ginkgo/extensions/kokkos/config.hpp.in
    ${Ginkgo_BINARY_DIR}/include/ginkgo/extensions/kokkos/config.hpp
    @ONLY
)
```

It substitutes the `GINKGO_EXTENSION_KOKKOS_CHECK_TYPE_ALIGNMENT` CMake
option into the generated `config.hpp`, so `value_type` can verify the
alignment of Ginkgo and Kokkos types at compile time.

## Anatomy of the config readers

The two config extensions turn a parsed file into a
`gko::config::pnode` property tree, which the runtime registry then
turns into solver and preconditioner factories.

- `config/json_config.hpp` — `gko::ext::config::parse_json`,
  `parse_json_file` and `parse_json_string`, built on `nlohmann::json`.
- `config/yaml_config.hpp` — `gko::ext::config::parse_yaml` and
  `parse_yaml_file`, built on yaml-cpp, including YAML anchors and
  `<<` merge keys.

Each header includes the third-party header directly, so a consumer's
build links nlohmann_json or yaml-cpp. Ginkgo's own tests link the
`yaml-cpp::yaml-cpp` and `nlohmann_json::nlohmann_json` targets.

## How extensions plug into the build

The top-level Ginkgo `CMakeLists.txt` calls
`add_subdirectory(extensions)` after the core build.
`extensions/CMakeLists.txt` adds the `test/` subdirectory when
`GINKGO_BUILD_TESTS=ON`. The per-extension tests are registered in
`extensions/test/CMakeLists.txt`, which discovers the third-party
dependency and adds the matching subdirectory:

```cmake
# extensions/CMakeLists.txt
if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()

# extensions/test/CMakeLists.txt
find_package(Kokkos 4.1.00 QUIET)
if(Kokkos_FOUND)
    add_subdirectory(kokkos)
endif()
add_subdirectory(config)
```

Header-only extensions build nothing themselves: consumers `#include`
the header, and Ginkgo's install step copies the `*.hpp` files of the
source and binary `include/` trees, which installs the generated Kokkos
`config.hpp` and leaves `config.hpp.in` in the source tree. The Kokkos
tests are skipped when Kokkos is missing.
`extensions/test/config/CMakeLists.txt` calls
`find_package(yaml-cpp 0.8.0 QUIET)` and fetches yaml-cpp with
`FetchContent` when it is missing; nlohmann_json is fetched from
`third_party/` in the same way.

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
- Add your test subdirectory to `extensions/test/CMakeLists.txt`, behind
  a `find_package(<MyExtDep> QUIET)` check as for Kokkos.
- In `extensions/test/<myext>/CMakeLists.txt`, link `Ginkgo::ginkgo`
  plus the third-party target through `ginkgo_create_test(...
  ADDITIONAL_LIBRARIES ...)`, as `extensions/test/config` does.

An extension that needs a compiled library would add its own
`extensions/<area>/` directory with a CMake target, link it publicly to
`ginkgo`, wire it from `extensions/CMakeLists.txt` and install it. That
layout is a contribution you would design together with the
maintainers.

## Design guidance: vendor-library wrappers

If you add a `LinOp` that wraps a vendor library, build it like any
other `LinOp` (see [Add a new solver](add-a-solver.md) or [Add a new
preconditioner](add-a-preconditioner.md) for the factory / parse
scaffolding) and choose its home by what it wraps:

- **In `core/`** — when the vendor path is one implementation among
  several for a type that Ginkgo already exposes. Example:
  `solver::LowerTrs` / `solver::UpperTrs` take an `algorithm`
  parameter. `trisolve_algorithm::sparselib` (the default) uses the
  vendor library (cuSPARSE on CUDA, hipSPARSE on HIP), and
  `trisolve_algorithm::syncfree` runs Ginkgo's own kernel on CUDA. The
  vendor calls live in `cuda/solver/lower_trs_kernels.cu` etc., gated by
  the executor type.
- **In `extensions/`** — when the integration is a bridge to a
  third-party library and not a Ginkgo type, as for the Kokkos and
  JSON/YAML config headers.

:::{seealso}
- [Add a new solver](add-a-solver.md) — LinOp + Factory + parse
  scaffolding that any vendor-backed solver also needs.
- [Add a new preconditioner](add-a-preconditioner.md) — same
  scaffolding, no workspace.
- [Configure via JSON](../using/configure-via-json.md) — what the JSON
  config extension unlocks for the runtime registry.
:::
