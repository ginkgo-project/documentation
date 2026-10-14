# Configure solvers via JSON

Instead of hard-coding solver factories in C++, you can describe them
in a JSON (or YAML) property tree and build the factory at runtime.
The same source tree can produce different solvers without
recompiling, which is useful for parameter sweeps, A/B testing, and
exposing solver choice to end users.

## The recipe

```cpp
using gko::config::pnode;

// The property tree — typically loaded from a JSON file.
auto config = pnode{pnode::map_type{
    {"type", pnode{"solver::Cg"}},
    {"criteria", pnode{pnode::array_type{
        pnode{pnode::map_type{{"type", pnode{"Iteration"}},
                              {"max_iters", pnode{1000}}}},
        pnode{pnode::map_type{{"type", pnode{"ResidualNorm"}},
                              {"reduction_factor", pnode{1e-8}}}}}}},
    {"preconditioner", pnode{pnode::map_type{
        {"type", pnode{"preconditioner::Jacobi"}}}}}}};
```

The registry knows which types the parser may build. Parsing returns a deferred factory
parameter, and `.on(exec)` turns it into a factory on the chosen executor:

```cpp
auto registry = gko::config::registry{};
auto factory = gko::config::parse(config, registry).on(exec);

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

The JSON reader in `ginkgo/extensions/config/json_config.hpp` parses a JSON file into a
`pnode` for you (see [Config file readers](../../../api-reference/extensions/config-readers.md)):

```cpp
#include <ginkgo/extensions/config/json_config.hpp>

auto config = gko::ext::config::parse_json_file("cg.json");
auto factory = gko::config::parse(config, registry).on(exec);
```

`parse_yaml_file` does the same for YAML. Both readers throw exceptions derived from
`std::exception` when the file cannot be read or parsed.

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

The `"type"` field names a Ginkgo factory type. The registered names are:

| Prefix | Type names |
|---|---|
| `solver::` | `Cg`, `Bicg`, `Bicgstab`, `Fcg`, `PipeCg`, `Cgs`, `Ir`, `Idr`, `Gcr`, `Gmres`, `CbGmres`, `Minres`, `Direct`, `LowerTrs`, `UpperTrs`, `Chebyshev`, `Multigrid` |
| `factorization::` | `Ic`, `Ilu`, `Cholesky`, `Lu`, `ParIc`, `ParIct`, `ParIlu`, `ParIlut` |
| `preconditioner::` | `GaussSeidel`, `Ic`, `Ilu`, `Isai`, `Jacobi`, `NullspaceProjection`, `Sor`, and `Schwarz` in MPI builds |
| `multigrid::` | `Pgm`, `Pmis`, `Rs`, `UniformCoarsening` |

Stopping criteria use the names `Iteration`, `ResidualNorm`, `ImplicitResidualNorm` and `Time`.
A multigrid level is selected by its type name, for example
`{"type": "multigrid::Pmis", "strength_threshold": 0.25}` as an `mg_level` entry.

### Singular systems

Iterative solvers that support nullspaces (`Cg`, `Fcg`, `Minres`, `Gmres` and `Bicgstab`) accept the
keys `"nullspace"` and `"left_nullspace"`. Each value is the name of an object stored in the
registry:

```cpp
registry.emplace("ns", gko::share(gko::solver::Nullspace<double>::create_from_constant(exec)));
```

```json
{"type": "solver::Cg", "nullspace": "ns", "criteria": {"iteration": 1000}}
```

See [Singular systems](../../concepts/solvers/singular-systems.md) for the nullspace
objects themselves.

## Register custom types

If your application defines its own `LinOpFactory` and wants to drive
it from config, supply a parse callable when constructing the
`registry`. The parse callable's signature is fixed by Ginkgo's
`configuration_map` typedef in
`include/ginkgo/core/config/registry.hpp`:

```cpp
using parse_fn = std::function<gko::deferred_factory_parameter<gko::LinOpFactory>(
    const gko::config::pnode&,
    const gko::config::registry&,
    gko::config::type_descriptor)>;
```

Wire it up like this:

```cpp
auto my_parse =
    [](const gko::config::pnode& config, const gko::config::registry& context,
       gko::config::type_descriptor td)
    -> gko::deferred_factory_parameter<gko::LinOpFactory> {
        // Read keys off `config`, look up nested factories in `context`,
        // and return a deferred factory parameter for your type. A Cg factory
        // stands in for your own LinOpFactory here.
        return gko::solver::Cg<double>::build().with_criteria(
            gko::stop::max_iters(config.get("max_iters").get_integer()));
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
- **Integers and reals.** `parse_json` turns JSON integers into integer
  nodes and numbers with a decimal point or exponent into real nodes;
  write counts such as `max_iters` as plain integers. Avoid quoting
  integers as strings.
- **Nested types.** Composing a Krylov solver with an ILU
  preconditioner means nesting `preconditioner` inside the solver's
  config; each nested object goes through the same `parse` path
  recursively.

:::{seealso}
- [Configuration from JSON files](../../concepts/config-files.md): the conceptual reference and full property-tree grammar.
- [`registry`](../../../api-reference/config/registry.md): the registry class and its type map.
:::
