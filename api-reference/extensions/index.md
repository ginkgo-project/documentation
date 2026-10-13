# `gko::ext`

Optional integrations with external libraries. The Kokkos and config-reader
headers are always installed and need their dependency only in your
application.

For the user-facing walk-through of each extension, see
[Extensions](../../user-guide/concepts/extensions.md).

## Available extensions

- Config file readers — `gko::ext::config::parse_json`, `parse_json_file`
  and `parse_json_string` (nlohmann_json), and `parse_yaml` and
  `parse_yaml_file` (yaml-cpp), turn a configuration file into the
  `gko::config::pnode` that `gko::config::parse` consumes. Their
  dependency is needed in your application's build.
- [`Kokkos interop`](kokkos.md) — helpers for round-tripping
  Ginkgo's `array`, `Dense`, and executor types through Kokkos views
  and execution spaces. Lives in `gko::ext::kokkos`; requires Kokkos
  in your application's build.

```{toctree}
:hidden:
:maxdepth: 1

kokkos
```
