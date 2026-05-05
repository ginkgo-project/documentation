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
gko::config::pnode root = gko::config::parse_json(json_string);
```

You typically do not construct `pnode` objects by hand — they are produced by a parser. The node tree captures the full structure of the config file and is independent of any particular Ginkgo type, making it easy to inspect or serialize back out.

## The registry

The registry maps "type name" strings (like `"solver::Cg"` or `"preconditioner::Jacobi"`) to factory builders. All built-in Ginkgo solver, preconditioner, and stopping-criterion types register themselves automatically.

```cpp
gko::config::registry r;
auto solver_factory = gko::config::parse(root, r,
                          gko::config::type_descriptor::create<double, int>())
                          .get_solver_factory();
```

The third argument, `type_descriptor`, carries the default `value_type` and `index_type` used to instantiate templated classes. Nodes that specify their own `"value_type"` or `"index_type"` in the JSON override the descriptor locally.

If you have a custom `LinOp` subclass you want to make configurable, you register it with the registry explicitly before calling `parse`. Without registration, a `"type"` string that names your class will produce an error at parse time rather than at compile time.

## The expected JSON format

A solver setup with a preconditioner and stopping criteria follows this schema:

```json
{
    "type": "solver::Cg",
    "value_type": "float64",
    "criteria": [
        {"type": "stop::Iteration", "max_iters": 1000},
        {"type": "stop::ResidualNorm",
         "value_type": "float64",
         "reduction_factor": 1e-8}
    ],
    "preconditioner": {
        "type": "preconditioner::Jacobi",
        "value_type": "float64",
        "index_type": "int32",
        "max_block_size": 8
    }
}
```

Three conventions govern this format:

- **`type` discriminator.** Every node that represents a Ginkgo object must carry a `"type"` key. The value is the unqualified class path within the Ginkgo namespace, e.g. `"solver::Cg"` for `gko::solver::Cg`.
- **Precision strings.** Templated types carry `"value_type"` (`"float32"`, `"float64"`, `"complex<float64>"`, …) and, where applicable, `"index_type"` (`"int32"`, `"int64"`). These strings must name types Ginkgo was compiled with.
- **Recursive composition.** Nested objects (criteria list, preconditioner, inner solver) follow the same schema recursively, matching the same [LinOp composition model](linop-and-composition.md) you use in C++.

## End-to-end example

```cpp
#include <ginkgo/extensions/config/json_config.hpp>

auto exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create());

gko::config::registry registry{};
auto root = gko::ext::config::parse_json_file("solver.json");
auto td   = gko::config::type_descriptor::create<double, gko::int32>();

auto solver_factory_lin = gko::config::parse(root, registry, td)
                              .get_solver_factory();

auto solver = std::dynamic_pointer_cast<gko::LinOpFactory>(solver_factory_lin)
                  ->generate(matrix);
solver->apply(b, x);
```

`parse_json_file` reads the file and returns a `pnode` tree. `gko::config::parse` walks that tree, looks up each `"type"` key in the registry, and recursively constructs the full factory hierarchy. The result's `get_solver_factory()` accessor returns a `std::shared_ptr` to the outermost factory — cast it to `gko::LinOpFactory` (or the specific type if you know it) to call `generate`.

:::{tip}
The solver factory produced this way is reusable. Call `generate(matrix)` multiple times with different matrices without re-parsing the JSON.
:::

## Limitations

- **Only registered types.** Custom `LinOp` subclasses are not constructible from config unless you register them with the registry first. The registration API is template-heavy and requires that the class expose a factory type.
- **Static type signatures.** The `"value_type"` and `"index_type"` strings must match types Ginkgo was compiled with. Requesting `"float128"` on a build that doesn't include it will fail at parse time.
- **No control flow.** Config is data — there are no conditionals, loops, or expressions. For dynamic solver selection, load different files from application code.
- **Extension header required.** File I/O and JSON parsing live in `ginkgo/extensions/config/json_config.hpp`, which is separate from the core `ginkgo/ginkgo.hpp`. Make sure your build links against the extensions target.

:::{seealso}
- [Solvers — taxonomy](solvers.md) — what you typically configure from a file.
- [Preconditioners — taxonomy](preconditioners.md) — what plugs into a solver.
- [LinOp and composition](linop-and-composition.md) — the same composition model in C++.
- API reference: [`gko::config`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/namespacegko_1_1config.html)
:::
