# Extensions — taxonomy

Ginkgo's `extensions/` tree holds optional integrations with third-party libraries. Each
extension is opt-in at build time via a CMake flag, and lives in its own namespace
under `gko::ext`. None of them are required by the core library; they exist so that
Ginkgo can plug into ecosystems where users already have data, executors, or solvers.
This page surveys the available extensions and points to their dedicated pages.

## What lives in `extensions/`

| Extension | Header | Namespace | Purpose |
|---|---|---|---|
| Kokkos interop | `<ginkgo/extensions/kokkos.hpp>` | `gko::ext::kokkos` | Wrap Kokkos execution spaces as Ginkgo executors and map Ginkgo containers to Kokkos `View`s. |
| cuDSS direct solver | `<ginkgo/extensions/cuda/solver/cudss.hpp>` | `gko::ext::cuda::solver` | Sparse direct solver wrapper around NVIDIA's cuDSS library. |
| JSON / YAML config | `<ginkgo/extensions/config/json_config.hpp>`, `<ginkgo/extensions/config/yaml_config.hpp>` | `gko::ext::config` | Parse solver configurations from JSON or YAML files. See [Configuration from JSON files](config-files.md) for the user-facing interface. |

The first two each have their own page below. The config extension is documented in
its own concept page because configuration is a first-class user-facing topic, not
just a build-time integration.

## Build-time enablement

Each extension requires the corresponding CMake flag to be set when Ginkgo is built:

| Extension | CMake flag |
|---|---|
| Kokkos interop | `-DGINKGO_EXTENSION_KOKKOS=ON` |
| cuDSS | `-DGINKGO_BUILD_EXT_CUDSS=ON` (also requires `-DGINKGO_BUILD_CUDA=ON`) |
| JSON / YAML config | `-DGINKGO_EXTENSION_CONFIG_BUILD_JSON=ON` / `-DGINKGO_EXTENSION_CONFIG_BUILD_YAML=ON` |

If the flag is off, the corresponding header is not installed and the extension's
namespace is not available.

## Per-extension pages

```{toctree}
:maxdepth: 1
:caption: Extensions

extensions/kokkos
extensions/cudss
```

:::{seealso}
- [Configuration from JSON files](config-files.md) — the user-facing wrapper around the config extension.
- API reference: [`gko::ext`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/namespacegko_1_1ext.html)
:::
