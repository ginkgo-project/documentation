# Configuration from JSON files

Ginkgo's `config` namespace lets an application load a complete solver or preconditioner stack from a JSON file at runtime, instead of hard-coding it in C++. This page covers the `pnode` in-memory representation, the registry of constructible types, the expected JSON format, and when file-based configuration makes sense.

## When to use file-based config

File-based config is the right tool when *runtime* selection of the solver setup matters:

- **Parameter sweeps.** Try fifty solver and preconditioner combinations without recompiling — write a loop over JSON files.
- **Application-supplied config.** End users choose a solver in a config file; your application code stays generic and recompile-free.
- **Reproducibility.** The exact solver setup is stored alongside results in a text file.

If your application always uses one fixed solver setup, the inline [factory builder pattern](linop-and-composition.md) is simpler and carries no overhead. Use file config when the point is flexibility at runtime.

## The pnode representation

A `gko::config::pnode` is the in-memory representation of a parsed config tree. Its structure is analogous to `nlohmann::json::value_type`: a node holds either a scalar (string, number, or boolean), an ordered array of child nodes, or an object (key/value map of child nodes).

```cpp
gko::config::pnode root = gko::ext::config::parse_json_string(json_string);
```

You typically do not construct `pnode` objects by hand — they are produced by a parser. The node tree captures the full structure of the config file and is independent of any particular Ginkgo type, making it easy to inspect or serialize back out.

## The registry

The registry maps "type name" strings (like `"solver::Cg"` or `"preconditioner::Jacobi"`) to factory builders. All built-in Ginkgo solver, preconditioner, and stopping-criterion types register themselves automatically.

```cpp
gko::config::registry r;
auto solver_factory =
    gko::config::parse(root, r, gko::config::make_type_descriptor<double, int>())
        .on(exec);
```

`parse` returns a deferred factory parameter; `.on(exec)` turns it into a `std::shared_ptr<gko::LinOpFactory>`. The third argument, `type_descriptor` (created with `make_type_descriptor<ValueType, IndexType>()`), carries the default `value_type` and `index_type` used to instantiate templated classes. Nodes that specify their own `"value_type"` in the JSON override the descriptor locally; the index type always comes from the descriptor.

If you have a custom `LinOp` subclass you want to make configurable, you register it with the registry explicitly before calling `parse`. Without registration, a `"type"` string that names your class will produce an error at parse time rather than at compile time.

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

- **`type` discriminator.** Every node that represents a Ginkgo object must carry a `"type"` key. The value is the unqualified class path within the Ginkgo namespace, e.g. `"solver::Cg"` for `gko::solver::Cg`.
- **Precision strings.** Templated types carry `"value_type"` (`"float32"`, `"float64"`, `"complex<float64>"`, …). The value string must name a type Ginkgo was compiled with. Index types are chosen through the `type_descriptor` passed to `parse` (`make_type_descriptor<ValueType, IndexType>()`).
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

`parse_json_file` reads the file and returns a `pnode` tree. `gko::config::parse` walks that tree, looks up each `"type"` key in the registry, and recursively builds the full factory hierarchy. Calling `.on(exec)` on the result yields the outermost `gko::LinOpFactory`, whose `generate(matrix)` produces the configured solver.

:::{tip}
The solver factory produced this way is reusable. Call `generate(matrix)` multiple times with different matrices without re-parsing the JSON.
:::

## Limitations

- **Only registered types.** Custom `LinOp` subclasses are not constructible from config unless you register them with the registry first. The registration API is template-heavy and requires that the class expose a factory type.
- **Static type signatures.** The `"value_type"` strings must match types Ginkgo was compiled with. Requesting `"float16"` on a build with `GINKGO_ENABLE_HALF=OFF` fails at parse time with a `gko::InvalidStateError` ("The provided runtime type … doesn't match any of the allowed compile time types"). The same error appears for `"float16"` and `"bfloat16"` on solvers parsed with the base type list only, such as `CbGmres`.
- **No control flow.** Config is data — there are no conditionals, loops, or expressions. For dynamic solver selection, load different files from application code.
- **Reader headers are header-only.** JSON parsing lives in `ginkgo/extensions/config/json_config.hpp` and YAML parsing in `ginkgo/extensions/config/yaml_config.hpp`; both are separate from `ginkgo/ginkgo.hpp` and are installed with Ginkgo. In your own CMake project, link `nlohmann_json::nlohmann_json` for JSON or `yaml-cpp::yaml-cpp` for YAML in addition to `Ginkgo::ginkgo`.

:::{seealso}
- [Solvers — taxonomy](solvers.md) — what you typically configure from a file.
- [Preconditioners — taxonomy](preconditioners.md) — what plugs into a solver.
- [LinOp and composition](linop-and-composition.md) — the same composition model in C++.
- API reference: [`gko::config`](../../api-reference/config/index.md)
:::
