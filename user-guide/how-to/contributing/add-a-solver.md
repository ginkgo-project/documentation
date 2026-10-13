# Add a new iterative solver

Ginkgo iterative solvers follow a strict CRTP skeleton: a class
template `Solver<ValueType>` that inherits from `EnableLinOp<Solver>`,
plumbs a workspace through `workspace_traits`, and runs its iteration
loop in `apply_dense_impl`. This page walks through the pattern using
`gko::solver::Cg` (`include/ginkgo/core/solver/cg.hpp`,
`core/solver/cg.cpp`) as the worked example.

## What you commit to implement

| File | What it holds |
|---|---|
| `include/ginkgo/core/solver/my_solver.hpp` | Class template, `parameters_type`, `workspace_traits` specialisation |
| `core/solver/my_solver.cpp` | `apply_impl`, `apply_dense_impl`, `transpose`/`conj_transpose`, workspace-trait method bodies, `parse` for JSON config |
| `core/solver/my_solver_kernels.hpp` | Per-step kernel declarations (e.g. CG has `initialize` + `step_1` + `step_2`) |
| One backend implementation per kernel | `reference/solver/`, `common/unified/solver/` or `common/cuda_hip/solver/`, `omp/solver/`, `dpcpp/solver/` |
| `core/config/config_helper.hpp` | Add an enum entry to `LinOpFactoryType` |
| `core/config/registry.cpp` | Add `"solver::MySolver"` to the configuration map |
| `core/config/solver_config.cpp` | Add the `GKO_PARSE_VALUE_TYPE(MySolver, gko::solver::MySolver)` specialisation |
| `core/CMakeLists.txt` | Add `solver/my_solver.cpp` to the source list |

Kernel files follow the same rules as in [Add a new kernel](add-a-kernel.md).

## The class skeleton

```cpp
template <typename ValueType = default_precision>
class MySolver
    : public EnableLinOp<MySolver<ValueType>>,
      public EnablePreconditionedIterativeSolver<ValueType, MySolver<ValueType>>,
      public Transposable {
    friend class EnableLinOp<MySolver>;
    friend class EnablePolymorphicObject<MySolver, LinOp>;
    GKO_ASSERT_SUPPORTED_VALUE_TYPE;

public:
    using value_type      = ValueType;
    using transposed_type = MySolver<ValueType>;     // self if symmetric

    std::unique_ptr<LinOp> transpose() const override;
    std::unique_ptr<LinOp> conj_transpose() const override;

    bool apply_uses_initial_guess() const override { return true; }

    class Factory;

    struct parameters_type
        : enable_preconditioned_iterative_solver_factory_parameters<
              parameters_type, Factory> {
        // Add extra solver-specific parameters here, e.g.:
        //     int GKO_FACTORY_PARAMETER_SCALAR(krylov_dim, 30);
    };

    GKO_ENABLE_LIN_OP_FACTORY(MySolver, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    static parameters_type parse(
        const config::pnode& config,
        const config::registry& context,
        const config::type_descriptor& td_for_child =
            config::make_type_descriptor<ValueType>());

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override;

    template <typename VectorType>
    void apply_dense_impl(const VectorType* b, VectorType* x) const;

    void apply_impl(const LinOp* alpha, const LinOp* b,
                    const LinOp* beta, LinOp* x) const override;

    explicit MySolver(std::shared_ptr<const Executor> exec)
        : EnableLinOp<MySolver>(std::move(exec)) {}

    explicit MySolver(const Factory* factory,
                      std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<MySolver>(factory->get_executor(),
                                gko::transpose(system_matrix->get_size())),
          EnablePreconditionedIterativeSolver<ValueType, MySolver>{
              std::move(system_matrix), factory->get_parameters()},
          parameters_{factory->get_parameters()} {}
};
```

Key pieces:

- `EnableLinOp<MySolver>` injects the public `apply` overloads — you only
  implement `apply_impl`.
- `EnablePreconditionedIterativeSolver` is a mixin that gives every
  preconditioned iterative solver `with_criteria(...)`,
  `with_preconditioner(...)`, `with_generated_preconditioner(...)`,
  plus the corresponding getters. Subclassing it pulls the factory
  parameter trait in via `enable_preconditioned_iterative_solver_factory_parameters`.
- `apply_uses_initial_guess() { return true; }` tells the framework that
  `x` on entry to `apply` is the initial guess and must not be zeroed.
- `GKO_ENABLE_LIN_OP_FACTORY` and `GKO_ENABLE_BUILD_METHOD` together
  define the `Factory` class, `parameters_`, the `build()` method, and
  the `with_*` setters. They're macro-heavy boilerplate; the
  `parameters_type` struct above is the one place you decide what new
  knobs the solver exposes.

## workspace_traits — the typed scratch pool

Iterative solvers allocate the same temporary vectors and scalars every
iteration. To avoid reallocating on every `apply`, the base
`SolverBaseLinOp` holds a `mutable workspace` that caches them. The
solver tells the workspace how many slots it needs, and what each slot
is for, through a `workspace_traits<MySolver<V>>` specialisation:

```cpp
template <typename ValueType>
struct workspace_traits<MySolver<ValueType>> {
    using Solver = MySolver<ValueType>;
    static int num_vectors(const Solver&);
    static int num_arrays(const Solver&);
    static std::vector<std::string> op_names(const Solver&);
    static std::vector<std::string> array_names(const Solver&);
    static std::vector<int> scalars(const Solver&);
    static std::vector<int> vectors(const Solver&);

    // Slot IDs for operators (Dense* vectors / scalars):
    constexpr static int r        = 0;  // residual
    constexpr static int z        = 1;  // preconditioned residual
    constexpr static int p        = 2;
    constexpr static int q        = 3;  // A*p
    constexpr static int beta     = 4;
    constexpr static int prev_rho = 5;
    constexpr static int rho      = 6;
    constexpr static int one      = 7;  // constant 1.0
    constexpr static int minus_one = 8; // constant -1.0

    // Slot IDs for typed arrays:
    constexpr static int stop = 0;  // stopping_status array
    constexpr static int tmp  = 1;  // reduction scratch (char)
};
```

The trait method bodies in `.cpp` just return constants and string
lists; e.g. CG returns `9` vectors and `2` arrays. The names appear in
log output, so make them descriptive.

`scalars()` returns the IDs of the slots that are `1×nrhs` (independent
of problem size), `vectors()` returns the IDs of the slots that are
`n×nrhs` (allocated per problem). The base class uses this to decide
when to reallocate and when to reuse.

## apply_impl and the precision dispatch

The two `apply_impl` overrides dispatch precision and then delegate to
the templated `apply_dense_impl`. Use the distributed-aware variant —
it accepts both `experimental::distributed::Vector` and `matrix::Dense`, so the same code
path works in serial and MPI builds:

```cpp
template <typename ValueType>
void MySolver<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    experimental::precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}

template <typename ValueType>
void MySolver<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                     const LinOp* beta, LinOp* x) const
{
    experimental::precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_dense_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone);
        },
        alpha, b, beta, x);
}
```

If your solver is not distributed-ready, use
`precision_dispatch_real_complex<ValueType>` instead — it accepts only
`matrix::Dense`.

## apply_dense_impl — the iteration loop

This is the actual algorithm. The boilerplate macros from
`core/solver/solver_boilerplate.hpp` allocate the workspace slots into
named local variables of type `Dense<V>*`:

```cpp
template <typename ValueType>
template <typename VectorType>
void MySolver<ValueType>::apply_dense_impl(const VectorType* dense_b,
                                           VectorType* dense_x) const
{
    using std::swap;
    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();
    this->setup_workspace();

    GKO_SOLVER_VECTOR(r, dense_b);
    GKO_SOLVER_VECTOR(z, dense_b);
    GKO_SOLVER_VECTOR(p, dense_b);
    GKO_SOLVER_VECTOR(q, dense_b);

    GKO_SOLVER_SCALAR(beta, dense_b);
    GKO_SOLVER_SCALAR(prev_rho, dense_b);
    GKO_SOLVER_SCALAR(rho, dense_b);

    GKO_SOLVER_ONE_MINUS_ONE();
    GKO_SOLVER_STOP_REDUCTION_ARRAYS();
    bool one_changed{};

    // r = b - A*x;  rho = 0;  prev_rho = 1;  z, p, q = 0
    exec->run(my_solver::make_initialize(
        gko::detail::get_local(dense_b), gko::detail::get_local(r),
        /* ... */ &stop_status));
    this->get_system_matrix()->apply(neg_one_op, dense_x, one_op, r);

    auto stop_criterion = this->get_stop_criterion_factory()->generate(
        this->get_system_matrix(),
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}),
        dense_x, r);

    int iter = -1;
    while (true) {
        this->get_preconditioner()->apply(r, z);
        r->compute_conj_dot(z, rho, reduction_tmp);

        ++iter;
        bool all_stopped =
            stop_criterion->update()
                .num_iterations(iter)
                .residual(r)
                .implicit_sq_residual_norm(rho)
                .solution(dense_x)
                .check(RelativeStoppingId, true,
                       &stop_status, &one_changed);
        this->template log<log::Logger::iteration_complete>(
            this, dense_b, dense_x, iter, r, nullptr, rho,
            &stop_status, all_stopped);
        if (all_stopped) {
            break;
        }

        // ... rest of the iteration steps as kernels ...
        swap(prev_rho, rho);
    }
}
```

Boilerplate macros (from `core/solver/solver_boilerplate.hpp`):

| Macro | Effect |
|---|---|
| `GKO_SOLVER_VECTOR(name, tmpl)` | Allocates a `Dense*` matching `tmpl`'s shape into a local `name`. |
| `GKO_SOLVER_SCALAR(name, tmpl)` | Allocates a `1×nrhs` `Dense*` (`ValueType`) into `name`. |
| `GKO_SOLVER_ONE_MINUS_ONE()` | Creates the constants `one_op = 1` and `neg_one_op = −1`. |
| `GKO_SOLVER_STOP_REDUCTION_ARRAYS()` | Allocates the `stop_status` and `reduction_tmp` `array<...>`s. |

`gko::detail::get_local(v)` is the internal helper (in
`core/distributed/helpers.hpp`) that Ginkgo's own solvers use to extract
the rank-local `matrix::Dense<V>*` from a vector pointer. It works for
both `experimental::distributed::Vector` and ordinary `matrix::Dense`,
so the same iteration body covers both cases.

## transpose, conj_transpose

When the solver is symmetric (CG, MINRES, Chebyshev), `transposed_type
= MySolver<ValueType>` and `transpose()` rebuilds the solver around the
transposed system matrix and transposed preconditioner. Copy the body
from `Cg::transpose` / `Cg::conj_transpose` in `core/solver/cg.cpp`
and adjust the operator factory. If your solver is not transposable,
do not include `public Transposable` in the base list.

## Parse and register for JSON config

To make the solver constructible from a `pnode` / JSON tree, three
edits are needed:

1. **`core/config/config_helper.hpp`** — append `MySolver` to the
   `LinOpFactoryType` enum.
2. **`core/config/registry.cpp`** — add an entry to
   `generate_config_map()`:
   ```cpp
   {"solver::MySolver", parse<LinOpFactoryType::MySolver>},
   ```
3. **`core/config/solver_config.cpp`** — add the parse specialisation:
   ```cpp
   GKO_PARSE_VALUE_TYPE(MySolver, gko::solver::MySolver);
   ```
   This expands into the explicit `template <> parse<LinOpFactoryType::MySolver>`
   specialisation. For a value-and-index-templated solver, use
   `GKO_PARSE_VALUE_AND_INDEX_TYPE`. The `Direct` solver and the
   triangular solvers are the existing precedents.

The corresponding `MySolver::parse(...)` body inside `core/solver/my_solver.cpp`
is one-liner — let `common_solver_parse` do the work:

```cpp
template <typename ValueType>
typename MySolver<ValueType>::parameters_type MySolver<ValueType>::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = MySolver<ValueType>::build();
    config::config_check_decorator config_check(config);
    config::common_solver_parse(params, config_check, context, td_for_child);
    return params;
}
```

If your solver has parameters beyond the preconditioned-iterative-solver
defaults, parse those out of `config_check` between the
`common_solver_parse` call and the return.

## Instantiation

End the `.cpp` with the value-type-explicit instantiation block:

```cpp
#define GKO_DECLARE_MY_SOLVER(ValueType)        class MySolver<ValueType>
#define GKO_DECLARE_MY_SOLVER_TRAITS(ValueType) \
    struct workspace_traits<MySolver<ValueType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MY_SOLVER);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MY_SOLVER_TRAITS);
```

Both lines are required — the second is what makes
`workspace_traits<MySolver<double>>::num_vectors(...)` actually link.

## Reference-parity test

The full test plan is covered in [Write tests](write-tests.md). For a
new solver you write three files:

- `core/test/solver/my_solver.cpp` — pure-C++ tests on the factory:
  `parameters_type` defaults, `with_*` setters, `build()` accepting
  generated preconditioners, `transpose()` returning a valid object.
  These don't need a system matrix or iteration. Registered with
  `ginkgo_create_test(my_solver)` in `core/test/solver/CMakeLists.txt`.
- `reference/test/solver/my_solver_kernels.cpp` — Reference-only
  convergence and correctness tests on a small SPD or general system
  (depending on the solver class). Registered with
  `ginkgo_create_test(my_solver_kernels)` in
  `reference/test/solver/CMakeLists.txt`.
- `test/solver/my_solver_kernels.cpp` — the **cross-backend parity
  test**. Same source file compiled once per enabled backend (`omp`,
  `cuda`, `hip`, `dpcpp`) by `ginkgo_create_common_test`, using
  `CommonTestFixture` (from `test/utils/common_fixture.hpp`) which
  provides both `ref` (Reference) and `exec` (the target backend).
  Each test sets up the same system on both, runs the solver, and
  asserts the solutions agree with `GKO_ASSERT_MTX_NEAR(...)`.
  Registered with `ginkgo_create_common_test(my_solver_kernels)` in
  `test/solver/CMakeLists.txt`.

`cg_kernels.cpp` in each of those three trees is the existing template
to copy from.

:::{seealso}
- [Add a new kernel](add-a-kernel.md) — for the iteration-step kernels.
- [Add a new preconditioner](add-a-preconditioner.md) — the simpler
  no-workspace variant of this pattern.
- [Configure via JSON](../using/configure-via-json.md) — what the
  registry edits above unlock for the end user.
:::
