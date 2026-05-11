# Configure solvers via JSON

Instead of hard-coding solver factories in C++, you can describe them
in a JSON (or YAML) property tree and build the factory at runtime.
The same source tree can produce different solvers without
recompiling, which is useful for parameter sweeps, A/B testing, and
exposing solver choice to end users.

## The recipe

```cpp
#include <ginkgo/ginkgo.hpp>

// The property tree — typically loaded from a JSON file.
auto config = gko::config::pnode{{
    {"type", gko::config::pnode{"solver::Cg"}},
    {"criteria", gko::config::pnode{{
        {{"type", gko::config::pnode{"Iteration"}},
         {"max_iters", gko::config::pnode{1000}}},
        {{"type", gko::config::pnode{"ResidualNorm"}},
         {"reduction_factor", gko::config::pnode{1e-8}}}
    }}},
    {"preconditioner", gko::config::pnode{{
        {"type", gko::config::pnode{"preconditioner::Jacobi"}}
    }}}
}};

// The registry — knows what types the parser is allowed to build.
auto registry = gko::config::registry{};

// Parse → deferred factory parameter → factory on the chosen executor.
auto factory = gko::config::parse(config, registry)
                   .on(exec);

// Use the factory as if it had been built statically.
auto solver = factory->generate(system_matrix);
solver->apply(b, x);
```

`pnode` is Ginkgo's executor-independent property tree. The shape
above corresponds to a JSON document like:

```json
{
    "type": "solver::Cg",
    "criteria": [
        {"type": "Iteration", "max_iters": 1000},
        {"type": "ResidualNorm", "reduction_factor": 1e-8}
    ],
    "preconditioner": {"type": "preconditioner::Jacobi"}
}
```

## Load from a JSON file

The `extensions::config::json_config` helper parses a JSON file into a
`pnode` for you:

```cpp
#include <ginkgo/extensions/config/json_config.hpp>

auto config = gko::ext::config::parse_json_file("cg.json");
auto factory = gko::config::parse(config, registry).on(exec);
```

`parse_yaml_file` does the same for YAML.

## Type names

The `"type"` field names a Ginkgo type. The parser recognises the
standard solvers / preconditioners / matrix formats / reorderings; see
`include/ginkgo/core/config/registry.hpp` for the canonical list.
Examples: `solver::Cg`, `solver::Gmres`, `preconditioner::Jacobi`,
`preconditioner::Ic`, `factorization::Ilu`, `matrix::Csr`.

## Register custom types

If your application defines its own `LinOp` and wants to drive it from
config, add a `parse` static method on the class and register it:

```cpp
auto registry = gko::config::registry{
    {"MyLinOp", MyLinOp::parse_function}
};
```

`parse_function` must have signature
`deferred_factory_parameter<LinOpFactory>(pnode, registry, type_descriptor)`.

## Common pitfalls

- **Precision.** The parser respects a `type_descriptor` (third arg to
  `parse`) for the value and index types. Pass an explicit
  `type_descriptor` if you want anything other than the defaults.
- **JSON numbers are doubles.** Integer fields like `max_iters` are
  read as numeric; the parser converts to `size_type` at registration
  time. Avoid quoting integers as strings.
- **Nested types.** Composing a Krylov solver with an ILU
  preconditioner means nesting `preconditioner` inside the solver's
  config; each nested object goes through the same `parse` path
  recursively.

:::{seealso}
- [Configuration from JSON files](../../concepts/config-files.md) — the conceptual reference and full property-tree grammar.
- [`registry`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1config_1_1registry.html) — the canonical type list.
:::
