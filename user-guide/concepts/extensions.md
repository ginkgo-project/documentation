# Extensions — taxonomy

Ginkgo's `extensions/` tree holds optional integrations with third-party libraries. Each
extension is a set of headers that you include where needed, and lives in its own namespace
under `gko::ext`. None of them are required by the core library; they exist so that
Ginkgo can plug into ecosystems where users already have data, executors, or solvers.
This page surveys the available extensions and points to their dedicated pages.

## What lives in `extensions/`

| Extension | Header | Namespace | Purpose |
|---|---|---|---|
| Kokkos interop | `<ginkgo/extensions/kokkos.hpp>` | `gko::ext::kokkos` | Wrap Kokkos execution spaces as Ginkgo executors and map Ginkgo containers to Kokkos `View`s. |
| JSON / YAML config | `<ginkgo/extensions/config/json_config.hpp>`, `<ginkgo/extensions/config/yaml_config.hpp>` | `gko::ext::config` | Parse solver configurations from JSON or YAML files. See [Configuration from JSON files](config-files.md) for the user-facing interface. |

Kokkos interop has its own page below. The config extension is documented in
its own concept page because configuration is a first-class user-facing topic, not
just a build-time integration.

## Using an extension in your project

The extensions are header-only and installed with Ginkgo's other headers. Ginkgo's own
build does not need the third-party library; your application links it next to
`Ginkgo::ginkgo`:

| Extension | Link in your CMake project |
|---|---|
| Kokkos interop | `Ginkgo::ginkgo` and `Kokkos::kokkos` (after `find_package(Kokkos)`). |
| JSON config | `Ginkgo::ginkgo` and `nlohmann_json::nlohmann_json`. |
| YAML config | `Ginkgo::ginkgo` and `yaml-cpp::yaml-cpp`. |

The Ginkgo CMake option `GINKGO_EXTENSION_KOKKOS_CHECK_TYPE_ALIGNMENT` (default `ON`)
makes the Kokkos type mapping check the alignment of the source and target types.

## Per-extension pages

```{toctree}
:maxdepth: 1
:caption: Extensions

extensions/kokkos
```

:::{seealso}
- [Configuration from JSON files](config-files.md) — the user-facing wrapper around the config extension.
- API reference: [`gko::ext`](../../api-reference/extensions/index.md)
:::
