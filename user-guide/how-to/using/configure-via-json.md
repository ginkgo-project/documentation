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

## Worked example: mixed-precision multigrid preconditioning CG

The simple recipe above shows the on-ramp, but the property-tree
format scales to arbitrarily nested solver stacks. The configuration
below — taken verbatim from `examples/file-config-solver/config/mixed-pgm-multigrid-cg.json`
in the Ginkgo source — drives a double-precision CG whose
preconditioner is a multigrid V-cycle that drops to single precision
on every level below the finest. The fine-level Pgm coarsening and IR
smoother stay at `double`; the second level, coarsest solver, and
their Pgm coarsening all run in `float32`.

```json
{
    "type": "solver::Cg",
    "preconditioner": {
        "type": "solver::Multigrid",
        "max_levels": 10,
        "min_coarse_rows": 2,
        "pre_smoother": [
            {
                "type": "solver::Ir",
                "relaxation_factor": 0.9,
                "solver": {
                    "type": "preconditioner::Jacobi",
                    "max_block_size": 1
                },
                "criteria": [{"type": "Iteration", "max_iters": 1}]
            },
            {
                "type": "solver::Ir",
                "value_type": "float32",
                "relaxation_factor": 0.9,
                "solver": {
                    "type": "preconditioner::Jacobi",
                    "max_block_size": 1
                },
                "criteria": [{"type": "Iteration", "max_iters": 1}]
            }
        ],
        "post_uses_pre": true,
        "mg_level": [
            {"type": "multigrid::Pgm", "deterministic": true},
            {"type": "multigrid::Pgm", "value_type": "float32",
             "deterministic": true}
        ],
        "coarsest_solver": {
            "type": "solver::Ir",
            "value_type": "float32",
            "relaxation_factor": 0.9,
            "solver": {
                "type": "preconditioner::Jacobi",
                "max_block_size": 1
            },
            "criteria": [{"type": "Iteration", "max_iters": 4}]
        },
        "default_initial_guess": "zero",
        "criteria": [{"type": "Iteration", "max_iters": 1}]
    },
    "criteria": [
        {"type": "Iteration", "max_iters": 100},
        {"type": "ResidualNorm", "reduction_factor": 1e-8,
         "baseline": "absolute"}
    ]
}
```

Things to notice:

- **Per-level lists.** `pre_smoother` and `mg_level` are JSON arrays.
  The first entry applies to the finest level, the second applies to
  every coarser level — that's the [vector-parameter selection rule
  documented for `Multigrid`](../../concepts/solvers/multigrid.md).
- **`value_type` overrides.** Setting `"value_type": "float32"` on any
  sub-object retypes that part of the hierarchy. The fine level stays
  `double` (the outer CG's value type); the coarse hierarchy works in
  `float32`, halving its memory and bandwidth.
- **`post_uses_pre: true`** reuses the pre-smoother definitions as
  post-smoothers, so a single per-level smoother list covers both.
- **One-iteration smoothers.** Each smoother runs a single
  Jacobi-preconditioned IR step per visit; the V-cycle's structure
  comes from the level-wise recursion, not from inner-loop counts.
- **Coarsest solver budget.** `max_iters: 4` on the coarsest level
  means we accept an approximate coarse correction rather than solving
  to convergence — typical for AMG as a Krylov preconditioner.

A matching C++ driver — load the file, parse it, call `apply` — is
shipped in `examples/file-config-solver/` and works without any
recompilation when the JSON changes.

## Type names

The `"type"` field names a Ginkgo type. The parser recognises the
standard solvers / preconditioners / matrix formats / reorderings; see
`include/ginkgo/core/config/registry.hpp` for the canonical list.
Examples: `solver::Cg`, `solver::Gmres`, `preconditioner::Jacobi`,
`preconditioner::Ic`, `factorization::Ilu`, `matrix::Csr`.

## Register custom types

If your application defines its own `LinOpFactory` and wants to drive
it from config, supply a parse callable when constructing the
`registry`. The parse callable's signature is fixed by Ginkgo's
`configuration_map` typedef in
`include/ginkgo/core/config/registry.hpp`:

```cpp
std::function<gko::deferred_factory_parameter<gko::LinOpFactory>(
    const gko::config::pnode&,
    const gko::config::registry&,
    gko::config::type_descriptor)>
```

Wire it up like this:

```cpp
auto my_parse =
    [](const gko::config::pnode& config,
       const gko::config::registry& context,
       gko::config::type_descriptor td) {
        // Read keys off `config`, look up nested factories in `context`,
        // and return a deferred factory parameter for your type.
        return MyLinOpFactory::build()
            .with_some_param(config.get("some_param").get_integer());
    };

// `usr::` prefix is the convention suggested by the registry doxygen
// to avoid colliding with Ginkgo's built-in type names.
auto registry = gko::config::registry{{{"usr::MyLinOp", my_parse}}};

auto factory = gko::config::parse(config, registry).on(exec);
```

The doubled braces are intentional: the outer pair constructs the
`configuration_map` argument, the inner pair is the initialiser list
of `{name, callable}` entries. Mismatching the callable signature will
fail to convert to `std::function<...>` and the compiler points at the
inner brace.

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
