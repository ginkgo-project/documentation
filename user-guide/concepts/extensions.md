# Extension taxonomy

Ginkgo's `extensions/` tree holds optional integrations with third-party libraries, each in
its own namespace under `gko::ext`. None of them are required by the core library; they exist
so that Ginkgo can plug into ecosystems where users already have data, executors, or solvers.
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

## Using the extensions

| Extension | How to use |
|---|---|
| Kokkos interop | Header-only, installed with Ginkgo. In your project, `find_package(Kokkos)` (4.1 or newer) and link `Kokkos::kokkos` together with `Ginkgo::ginkgo`. |
| cuDSS | Built when Ginkgo is configured with `-DGINKGO_BUILD_CUDA=ON` and `find_package(cudss 0.8.0 CONFIG)` finds cuDSS (set `cudss_DIR` or `CMAKE_PREFIX_PATH`). Link `Ginkgo::ginkgo_cudss`. |
| JSON / YAML config | Header-only, installed with Ginkgo. Link `nlohmann_json::nlohmann_json` (JSON) or `yaml-cpp::yaml-cpp` (YAML) in your project. |

## Per-extension pages

```{toctree}
:maxdepth: 1
:caption: Extensions

extensions/kokkos
extensions/cudss
```

:::{seealso}
- [Configuration from JSON files](config-files.md): the user-facing wrapper around the config extension.
- API reference: [Config file readers](../../api-reference/extensions/config-readers.md), [`gko::ext`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/namespacegko_1_1ext.html)
:::
