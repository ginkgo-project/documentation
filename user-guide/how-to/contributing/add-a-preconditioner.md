# Add a new preconditioner

A preconditioner in Ginkgo is a `LinOp` with two extra entry points:
`generate(system_matrix)` which builds the operator from a source matrix,
and `apply(b, x)` which applies the resulting operator. There is no
workspace machinery — preconditioners do their work in a single
`apply` and have no iteration loop.

This page covers two patterns: a **local** preconditioner modelled on
`preconditioner::Jacobi` (`include/ginkgo/core/preconditioner/jacobi.hpp`,
`core/preconditioner/jacobi.cpp`), and a **distributed** preconditioner
modelled on `experimental::distributed::preconditioner::Schwarz`
(`include/ginkgo/core/distributed/preconditioner/schwarz.hpp`,
`core/distributed/preconditioner/schwarz.cpp`).

## What you commit to implement

| File | Local (Jacobi-style) | Distributed (Schwarz-style) |
|---|---|---|
| Public header | `include/ginkgo/core/preconditioner/my_precond.hpp` | `include/ginkgo/core/distributed/preconditioner/my_precond.hpp` |
| Implementation | `core/preconditioner/my_precond.cpp` | `core/distributed/preconditioner/my_precond.cpp` |
| Kernel header | `core/preconditioner/my_precond_kernels.hpp` | Usually reuses local kernels — no MPI in kernels |
| Backend kernels | `reference/preconditioner/`, `cuda/preconditioner/`, ... | Same |
| `core/CMakeLists.txt` | Add `preconditioner/my_precond.cpp` | Add under the `if(GINKGO_BUILD_MPI)` block |
| `core/config/config_helper.hpp` | Append to `LinOpFactoryType` | Append to `LinOpFactoryType` |
| `core/config/registry.cpp` | Add to `generate_config_map()` | Same, but guarded `#if GINKGO_BUILD_MPI` |
| `core/config/preconditioner_config.cpp` *(local)* or `core/config/indextype_dispatch.cpp` *(distributed)* | Parse specialisation | Parse specialisation |

## The local preconditioner skeleton

```cpp
template <typename ValueType = default_precision, typename IndexType = int32>
class MyPrecond
    : public EnableLinOp<MyPrecond<ValueType, IndexType>>,
      public Transposable {
    friend class EnableLinOp<MyPrecond>;
    friend class EnablePolymorphicObject<MyPrecond, LinOp>;
    GKO_ASSERT_SUPPORTED_VALUE_AND_INDEX_TYPE;

public:
    using value_type      = ValueType;
    using index_type      = IndexType;
    using transposed_type = MyPrecond<ValueType, IndexType>;

    std::unique_ptr<LinOp> transpose() const override;
    std::unique_ptr<LinOp> conj_transpose() const override;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        // Whatever the preconditioner needs to know at generate time:
        bool   GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, false);
        uint32 GKO_FACTORY_PARAMETER_SCALAR(max_block_size, 32u);
        // Sub-factories use the deferred variant:
        std::shared_ptr<const LinOpFactory>
            GKO_DEFERRED_FACTORY_PARAMETER(inner_factory);
    };
    GKO_ENABLE_LIN_OP_FACTORY(MyPrecond, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    static parameters_type parse(
        const config::pnode& config,
        const config::registry& context,
        const config::type_descriptor& td_for_child =
            config::make_type_descriptor<ValueType, IndexType>());

protected:
    explicit MyPrecond(std::shared_ptr<const Executor> exec)
        : EnableLinOp<MyPrecond>(std::move(exec)) {}

    explicit MyPrecond(const Factory* factory,
                       std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<MyPrecond>(factory->get_executor(),
                                 gko::transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()}
    {
        this->generate(system_matrix.get(), parameters_.skip_sorting);
    }

    void apply_impl(const LinOp* b, LinOp* x) const override;
    void apply_impl(const LinOp* alpha, const LinOp* b,
                    const LinOp* beta, LinOp* x) const override;

private:
    void generate(const LinOp* system_matrix, bool skip_sorting);

    // Whatever device-side state your generate() builds — e.g. a Csr
    // of factor entries, an array of block pointers, ...:
    array<value_type> data_;
};
```

Compared to a solver:

- No `EnablePreconditionedIterativeSolver` base — preconditioners don't
  iterate. The only base is `EnableLinOp<Self>`.
- No `workspace_traits` specialisation — there's no iteration scratch to
  cache.
- `generate(...)` runs from the second constructor (the one the factory
  uses). After it returns, the `private` state holds everything `apply`
  needs.

## generate

`generate` is preconditioner-specific. It typically:

1. Allocates the device-side storage for the preconditioner data (e.g.
   block diagonal blocks, ILU factor `L` and `U` matrices).
2. Optionally sorts the system matrix (`skip_sorting` lets callers skip
   this when they already know it's sorted).
3. Runs one or more kernels to populate that storage.

Use `gko::clone(this->get_executor(), as<...>(system_matrix))` if you
need a typed view of the input on the right executor.

## apply_impl

`apply_impl` for a local preconditioner uses the non-distributed
precision dispatch and runs a kernel that consumes the precomputed
state:

```cpp
template <typename ValueType, typename IndexType>
void MyPrecond<ValueType, IndexType>::apply_impl(const LinOp* b,
                                                 LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->get_executor()->run(my_precond::make_apply(
                /* preconditioner state */ data_,
                dense_b->get_const_device_view(),
                dense_x->get_device_view()));
        },
        b, x);
}

template <typename ValueType, typename IndexType>
void MyPrecond<ValueType, IndexType>::apply_impl(const LinOp* alpha,
                                                 const LinOp* b,
                                                 const LinOp* beta,
                                                 LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b,
               auto dense_beta, auto dense_x) {
            this->get_executor()->run(my_precond::make_apply(
                dense_alpha->get_const_device_view(),
                data_,
                dense_b->get_const_device_view(),
                dense_beta->get_const_device_view(),
                dense_x->get_device_view()));
        },
        alpha, b, beta, x);
}
```

## The distributed preconditioner skeleton

```cpp
template <typename ValueType = default_precision,
          typename LocalIndexType = int32,
          typename GlobalIndexType = int64>
class MyPrecond
    : public EnableLinOp<MyPrecond<ValueType, LocalIndexType, GlobalIndexType>> {
    friend class EnableLinOp<MyPrecond>;
    friend class EnablePolymorphicObject<MyPrecond, LinOp>;
    GKO_ASSERT_SUPPORTED_VALUE_AND_DIST_INDEX_TYPE;

public:
    using value_type        = ValueType;
    using local_index_type  = LocalIndexType;
    using global_index_type = GlobalIndexType;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        std::shared_ptr<const LinOpFactory>
            GKO_DEFERRED_FACTORY_PARAMETER(local_solver);
    };
    GKO_ENABLE_LIN_OP_FACTORY(MyPrecond, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    static parameters_type parse(
        const config::pnode& config,
        const config::registry& context,
        const config::type_descriptor& td_for_child =
            config::make_type_descriptor<ValueType, LocalIndexType,
                                         GlobalIndexType>());

protected:
    explicit MyPrecond(std::shared_ptr<const Executor> exec)
        : EnableLinOp<MyPrecond>(std::move(exec)) {}

    explicit MyPrecond(const Factory* factory,
                       std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<MyPrecond>(factory->get_executor(),
                                 gko::transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()}
    {
        this->generate(std::move(system_matrix));
    }

    void apply_impl(const LinOp* b, LinOp* x) const override;
    template <typename VectorType>
    void apply_dense_impl(const VectorType* b, VectorType* x) const;
    void apply_impl(const LinOp* alpha, const LinOp* b,
                    const LinOp* beta, LinOp* x) const override;

private:
    void generate(std::shared_ptr<const LinOp> system_matrix);

    std::shared_ptr<const LinOp> local_solver_;
    detail::VectorCache<ValueType> cache_;   // for advanced apply
};
```

Two differences from the local case:

- The class is templated on three types (`V`, `L`, `G`) and lives under
  `gko::experimental::distributed::preconditioner`. The header is wrapped
  in `#if GINKGO_BUILD_MPI` and the source is added to `core/CMakeLists.txt`
  under the `if(GINKGO_BUILD_MPI)` block.
- The static-assert is `GKO_ASSERT_SUPPORTED_VALUE_AND_DIST_INDEX_TYPE`,
  which constrains `LocalIndexType` and `GlobalIndexType` to the
  distributed-supported combinations.

The `apply_impl` for a purely local Schwarz-style preconditioner applies
the inner solver to the rank-local Dense slice and returns — there is no
MPI communication in the preconditioner itself:

```cpp
template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void MyPrecond<ValueType, LocalIndexType, GlobalIndexType>::apply_impl(
    const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}

template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
template <typename VectorType>
void MyPrecond<ValueType, LocalIndexType, GlobalIndexType>::apply_dense_impl(
    const VectorType* dense_b, VectorType* dense_x) const
{
    if (this->local_solver_) {
        this->local_solver_->apply(gko::detail::get_local(dense_b),
                                   gko::detail::get_local(dense_x));
    }
}
```

`gko::detail::get_local(v)` is the helper from
`core/distributed/helpers.hpp`. It returns the rank-local
`matrix::Dense<V>*` for both `distributed::Vector` and ordinary
`matrix::Dense`, so the same iteration body covers serial and MPI use.

The communication this preconditioner *needs* (typically a halo exchange
before the local solve) is the responsibility of the surrounding solver,
not of the preconditioner. The Schwarz preconditioner does no MPI
communication inside its own `apply_impl`.

## Config / JSON registration

For a local preconditioner, the parse specialisation lives in
`core/config/preconditioner_config.cpp` next to the existing
`GKO_PARSE_VALUE_AND_INDEX_TYPE(Jacobi, gko::preconditioner::Jacobi)`
line. For a value-only preconditioner, use `GKO_PARSE_VALUE_TYPE`.

For a distributed preconditioner, the equivalent lives in
`core/config/indextype_dispatch.cpp` (the Schwarz parse specialisation
is the existing precedent).

Then add the enum entry to `LinOpFactoryType` in
`core/config/config_helper.hpp` and the map entry to
`generate_config_map()` in `core/config/registry.cpp`. The Schwarz
example shows the MPI guard pattern:

```cpp
#if GINKGO_BUILD_MPI
    {"preconditioner::Schwarz", parse<LinOpFactoryType::Schwarz>},
#endif
```

## Instantiation

End the `.cpp` with the explicit-instantiation block. For a local
preconditioner:

```cpp
#define GKO_DECLARE_MY_PRECOND(ValueType, IndexType) \
    class MyPrecond<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_MY_PRECOND);
```

For a distributed preconditioner:

```cpp
#define GKO_DECLARE_MY_PRECOND(V, L, G) class MyPrecond<V, L, G>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(GKO_DECLARE_MY_PRECOND);
```

## Reference-parity test

The test pattern is the same as for matrix formats and solvers — a
small hand-checkable input, run on Reference and on the target backend,
compared within a tolerance. Existing tests:
`reference/test/preconditioner/jacobi_kernels.cpp` for the local case,
`test/mpi/distributed/preconditioner/schwarz.cpp` for the distributed
case. See [Write tests](write-tests.md) for the general convention.

:::{seealso}
- [Add a new solver](add-a-solver.md) — the iterative pattern, with
  workspace machinery.
- [Distributed solvers and preconditioners](../../concepts/distributed/solvers-and-preconditioners.md) — what the distributed
  preconditioner composes with.
- [Configure via JSON](../using/configure-via-json.md) — how config edits
  surface in user-facing JSON.
:::
