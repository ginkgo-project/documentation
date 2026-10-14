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
using gko::config::pnode;
auto config = pnode{pnode::map_type{
    {"type", pnode{"solver::Cg"}},
    {"criteria", pnode{pnode::array_type{
        pnode{pnode::map_type{{"type", pnode{"Iteration"}},
                              {"max_iters", pnode{1000}}}},
        pnode{pnode::map_type{{"type", pnode{"ResidualNorm"}},
                              {"reduction_factor", pnode{1e-8}}}}}}},
    {"preconditioner", pnode{pnode::map_type{
        {"type", pnode{"preconditioner::Jacobi"}}}}}}};

// The registry — knows what types the parser is allowed to build.
auto registry = gko::config::registry{};

// Parse → deferred factory parameter → factory on the chosen executor.
auto factory = gko::config::parse(config, registry)
                   .on(exec);

// Use the factory as if it had been built statically.
auto solver = factory->generate(system_matrix);
solver->apply(b, x);
```

`pnode` is Ginkgo's executor-independent property tree; a map node is
built from a `pnode::map_type` and a list node from a `pnode::array_type`.
The shape above corresponds to a JSON document like:

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

The `gko::ext::config::parse_json_file` helper from
`json_config.hpp` parses a JSON file into a `pnode` for you:

```cpp
#include <ginkgo/extensions/config/json_config.hpp>

auto config = gko::ext::config::parse_json_file("cg.json");
auto factory = gko::config::parse(config, registry).on(exec);
```

`parse_yaml_file` does the same for YAML.

## Worked example: mixed-precision multigrid preconditioning CG

The simple recipe above shows the on-ramp, but the property-tree
format scales to arbitrarily nested solver stacks. The configuration
below (taken verbatim from `examples/file-config-solver/config/mixed-pgm-multigrid-cg.json`
in the Ginkgo source) drives a double-precision CG whose
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
  every coarser level; that's the [vector-parameter selection rule
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
  to convergence, typical for AMG as a Krylov preconditioner.

A matching C++ driver (load the file, parse it, call `apply`) is
shipped in `examples/file-config-solver/` and works without any
recompilation when the JSON changes.

## Type names

The `"type"` field names a registered Ginkgo type: the `solver::`,
`factorization::` and `preconditioner::` types, plus `multigrid::Pgm`
for the coarsening of `solver::Multigrid`. Examples: `solver::Cg`,
`solver::Gmres`, `preconditioner::Jacobi`, `preconditioner::Ic`,
`factorization::Ilu`, `multigrid::Pgm`.

Entries of a `criteria` list use the stopping-criterion names `Iteration`,
`ResidualNorm`, `ImplicitResidualNorm` and `Time`.

## Register custom types

If your application defines its own `LinOpFactory` and wants to drive
it from config, supply a parse callable when constructing the
`registry`. The parse callable's signature is fixed by Ginkgo's
`configuration_map` typedef in
`include/ginkgo/core/config/registry.hpp`:

```cpp
using parse_fn = std::function<
    gko::deferred_factory_parameter<gko::LinOpFactory>(
        const gko::config::pnode&,
        const gko::config::registry&,
        gko::config::type_descriptor)>;
```

Wire it up like this:

```cpp
auto my_parse =
    [](const gko::config::pnode& config,
       const gko::config::registry& context,
       gko::config::type_descriptor td)
        -> gko::deferred_factory_parameter<gko::LinOpFactory> {
        // Read keys off `config`, look up nested factories in `context`,
        // and return the factory parameters of your type. Cg stands in
        // for your own factory's build().
        return gko::solver::Cg<double>::build().with_criteria(
            gko::stop::Iteration::build().with_max_iters(
                static_cast<gko::size_type>(
                    config.get("max_iters").get_integer())));
    };

// The `usr::` prefix keeps your names clear of Ginkgo's built-in types.
auto registry = gko::config::registry{
    gko::config::configuration_map{{"usr::MyLinOp", my_parse}}};

auto factory = gko::config::parse(config, registry).on(exec);
```

Three levels of braces nest here: the outermost pair initialises the
`registry`, the pair after `configuration_map` is its list of entries, and
the innermost pair is one `{name, callable}` entry. A callable whose signature differs
from `parse_fn` fails to convert to the `std::function` and the compiler
points at the inner brace.

## Common pitfalls

- **Precision.** The parser respects a `type_descriptor` (third arg to
  `parse`) for the value and index types. Pass an explicit
  `type_descriptor` if you want anything other than the defaults.
- **Integers and reals are distinct nodes.** `parse_json` turns JSON
  integers into integer nodes and numbers with a decimal point or
  exponent into real nodes. Write counts such as `max_iters` as plain
  integers, without a decimal point or quotes.
- **Nested types.** Composing a Krylov solver with an ILU
  preconditioner means nesting `preconditioner` inside the solver's
  config; each nested object goes through the same `parse` path
  recursively.

:::{seealso}
- [Configuration from JSON files](../../concepts/config-files.md): the conceptual reference and full property-tree grammar.
- [`registry`](../../../api-reference/config/registry.md): the canonical type list.
:::
