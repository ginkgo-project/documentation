# Configuration from JSON files

Ginkgo's `config` namespace lets an application load a complete solver or preconditioner stack from a JSON file at runtime, instead of hard-coding it in C++. This page covers the `pnode` in-memory representation, the registry of constructible types, the expected JSON format, and when file-based configuration makes sense.

## When to use file-based config

File-based config is the right tool when *runtime* selection of the solver setup matters:

- **Parameter sweeps.** Try fifty solver and preconditioner combinations without recompiling: write a loop over JSON files.
- **Application-supplied config.** End users choose a solver in a config file; your application code stays generic and recompile-free.
- **Reproducibility.** The exact solver setup is stored alongside results in a text file.

If your application always uses one fixed solver setup, the inline [factory builder pattern](linop-and-composition.md) is simpler and carries no overhead. Use file config when the point is flexibility at runtime.

## The pnode representation

A `gko::config::pnode` is the in-memory representation of a parsed config tree. Its structure is analogous to `nlohmann::json::value_type`: a node holds either a scalar (string, number, or boolean), an ordered array of child nodes, or an object (key/value map of child nodes).

```cpp
#include <ginkgo/extensions/config/json_config.hpp>

gko::config::pnode root = gko::ext::config::parse_json_string(json_string);
```

The readers are described in [Config file readers](../../api-reference/extensions/config-readers.md). You typically do not construct `pnode` objects by hand; they are produced by a parser. The node tree captures the full structure of the config file and is independent of any particular Ginkgo type, making it easy to inspect or serialize back out.

## The registry

The registry maps "type name" strings (like `"solver::Cg"` or `"preconditioner::Jacobi"`) to factory builders. All built-in Ginkgo solver, preconditioner, and stopping-criterion types register themselves automatically.

```cpp
gko::config::registry r;
auto solver_factory = gko::config::parse(
                          root, r, gko::config::make_type_descriptor<double, int>())
                          .on(exec);
```

`gko::config::parse` returns a deferred factory; `.on(exec)` turns it into a `std::shared_ptr<gko::LinOpFactory>` on `exec`, ready for `generate`. The third argument, a `type_descriptor` created with `gko::config::make_type_descriptor<V, I>()`, carries the default `value_type` and `index_type` used to instantiate templated classes. Nodes that specify their own `"value_type"` in the JSON override the descriptor locally; the index type always comes from the `type_descriptor`.

Additional types are added by passing a `gko::config::configuration_map` (type name to parse function) to the registry constructor. A `"type"` string that names an unregistered class produces an error at parse time.

### Available types

| Prefix | Type names |
|---|---|
| `solver::` | `Cg`, `Bicg`, `Bicgstab`, `Fcg`, `PipeCg`, `Cgs`, `Ir`, `Idr`, `Gcr`, `Gmres`, `CbGmres`, `Minres`, `Direct`, `LowerTrs`, `UpperTrs`, `Chebyshev`, `Multigrid` |
| `factorization::` | `Ic`, `Ilu`, `Cholesky`, `Lu`, `ParIc`, `ParIct`, `ParIlu`, `ParIlut` |
| `preconditioner::` | `GaussSeidel`, `Ic`, `Ilu`, `Isai`, `Jacobi`, `Sor`, `NullspaceProjection`, and `Schwarz` in MPI builds |
| `multigrid::` | `Pgm`, `Pmis`, `Rs`, `UniformCoarsening` |

The solvers that support nullspaces (`Cg`, `Fcg`, `Minres`, `Gmres` and `Bicgstab`) accept `"nullspace"` and `"left_nullspace"`: the name of a `gko::solver::Nullspace` object that you stored in the registry with `registry.emplace("my_nullspace", nullspace)`.

The cuDSS extension provides a configuration map of its own; see [cuDSS direct solver](extensions/cudss.md#configuration-files).

## The expected JSON format

A solver setup with a preconditioner and stopping criteria follows this schema:

```json
{
    "type": "solver::Cg",
    "value_type": "float64",
    "criteria": [
        {"type": "Iteration", "max_iters": 1000},
        {"type": "ResidualNorm",
         "value_type": "float64",
         "reduction_factor": 1e-8}
    ],
    "preconditioner": {
        "type": "preconditioner::Jacobi",
        "value_type": "float64",
        "max_block_size": 8
    }
}
```

Three conventions govern this format:

- **`type` discriminator.** Every node that represents a Ginkgo object must carry a `"type"` key. The value is the unqualified class path within the Ginkgo namespace, e.g. `"solver::Cg"` for `gko::solver::Cg`. The `gko::`, `experimental::` and `stop::` prefixes are omitted, e.g. `"Iteration"` for `gko::stop::Iteration`.
- **Precision strings.** Templated types carry `"value_type"` (`"float32"`, `"float64"`, `"complex<float64>"`, …). Index types come from `make_type_descriptor<V, I>()` (for example `int32` or `int64`). The `"value_type"` strings must name types Ginkgo was compiled with.
- **Recursive composition.** Nested objects (criteria list, preconditioner, inner solver) follow the same schema recursively, matching the same [LinOp composition model](linop-and-composition.md) you use in C++.

## End-to-end example

```cpp
#include <ginkgo/extensions/config/json_config.hpp>

auto exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create());

gko::config::registry registry{};
auto root = gko::ext::config::parse_json_file("solver.json");
auto td   = gko::config::make_type_descriptor<double, gko::int32>();

auto solver_factory = gko::config::parse(root, registry, td).on(exec);
auto solver = solver_factory->generate(matrix);
solver->apply(b, x);
```

`parse_json_file` reads the file and returns a `pnode` tree. `gko::config::parse` walks that tree, looks up each `"type"` key in the registry, and recursively constructs the full factory hierarchy. `.on(exec)` on the result returns a `std::shared_ptr<gko::LinOpFactory>` for the outermost factory, on which you call `generate`.

Instead of a `"criteria"` array, stopping criteria can be given in a short form, for example `"criteria": {"iteration": 1000, "relative_residual_norm": 1e-8}`.

:::{tip}
The solver factory produced this way is reusable. Call `generate(matrix)` multiple times with different matrices without re-parsing the JSON.
:::

## Limitations

- **Only registered types.** Custom `LinOp` subclasses are constructible from config once you add them to the registry through a `configuration_map`. The class must expose a factory type.
- **Static type signatures.** The `"value_type"` strings must match types Ginkgo was compiled with. Requesting `"float16"` on a build with `GINKGO_ENABLE_HALF=OFF` fails at parse time with a `gko::InvalidStateError`.
- **Config is data.** A file holds values only, with no conditionals, loops, or expressions. For dynamic solver selection, load different files from application code.
- **Parser library required.** The JSON and YAML readers are header-only (`<ginkgo/extensions/config/json_config.hpp>` and `yaml_config.hpp`) and separate from `ginkgo/ginkgo.hpp`. Link your target against `nlohmann_json::nlohmann_json` (JSON) or `yaml-cpp::yaml-cpp` (YAML) alongside `Ginkgo::ginkgo`. Failures while reading or parsing throw an exception derived from `std::exception`.

:::{seealso}
- [Solver taxonomy](solvers.md): what you typically configure from a file.
- [Preconditioner taxonomy](preconditioners.md): what plugs into a solver.
- [LinOp and composition](linop-and-composition.md): the same composition model in C++.
- API reference: [`gko::config`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/namespacegko_1_1config.html)
:::
