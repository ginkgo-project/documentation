# Add a new preconditioner

A preconditioner in Ginkgo is a `LinOp` with two extra entry points:
`generate(system_matrix)` which builds the operator from a source matrix,
and `apply(b, x)` which applies the resulting operator. Preconditioners
do their work in a single `apply`, so they need neither workspace
machinery nor an iteration loop.

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
| Backend kernels | `reference/preconditioner/`, `common/cuda_hip/preconditioner/`, ... | Same |
| `core/CMakeLists.txt` | Add `preconditioner/my_precond.cpp` | Add under the `if(GINKGO_BUILD_MPI)` block |
| `core/config/config_helper.hpp` | Append to `LinOpFactoryType` | Append to `LinOpFactoryType` |
| `core/config/registry.cpp` | Add to `generate_config_map()` | Same, but guarded `#if GINKGO_BUILD_MPI` |
| Parse specialisation | A line in `core/config/preconditioner_config.cpp` (or its own file when dispatch is non-trivial) | A new file `core/config/my_precond_config.cpp`, wired in under the `if(GINKGO_BUILD_MPI)` block of `core/CMakeLists.txt` |

## The local preconditioner skeleton

The local classes live in `namespace gko::preconditioner`.

```cpp
template <typename ValueType = default_precision, typename IndexType = int32>
class MyPrecond
    : public LinOp,
      public Transposable {
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
        : LinOp(std::move(exec)) {}

    explicit MyPrecond(const Factory* factory,
                       std::shared_ptr<const LinOp> system_matrix)
        : LinOp(factory->get_executor(),
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
  iterate. The only base is `LinOp` (plus `Transposable` if the
  preconditioner can be transposed). `Ilu`, `Ic`, `Isai` and `Schwarz`
  are plain `LinOp` classes of this kind.
- To make the preconditioner cloneable (`gko::clone`, `copy_from`), also
  derive from `EnableCloneable<MyPrecond<ValueType, IndexType>>`, add
  `friend class EnableCloneable<MyPrecond<ValueType, IndexType>>;`, and
  declare the copy and move constructors and assignment operators, as
  `Jacobi` does. See [Cloneable](../../../api-reference/base/cloneable.md).
- Preconditioners skip `workspace_traits`: all state is built once in
  `generate`.
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
                data_,
                dense_alpha->get_const_device_view(),
                dense_b->get_const_device_view(),
                dense_beta->get_const_device_view(),
                dense_x->get_device_view()));
        },
        alpha, b, beta, x);
}
```

### Reusing setup across same-pattern matrices (optional)

:::{attention}
`generate_reuse` is an experimental interface and may change.
:::

When users solve a sequence of matrices with one sparsity pattern, the
preconditioner can keep what it computed for the first matrix.
`multigrid::Pgm` implements this and is the precedent to copy. Replace
`GKO_ENABLE_LIN_OP_FACTORY` with `GKO_ENABLE_LIN_OP_FACTORY_WITH_REUSE`,
naming a `reuse_data_type` that is declared before the macro and defined
after it:

```cpp
template <typename ValueType = default_precision, typename IndexType = int32>
class MyPrecond : public LinOp {
public:
    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory){};
    class reuse_data_type;
    GKO_ENABLE_LIN_OP_FACTORY_WITH_REUSE(MyPrecond, parameters, Factory,
                                         reuse_data_type);
    GKO_ENABLE_BUILD_METHOD(Factory);
    class reuse_data_type : public LinOpFactory::ReuseData {
        friend class MyPrecond;
        bool initialized_ = false;
        array<IndexType> pattern_;  // computed by the first generation
    };
protected:
    void apply_impl(const LinOp* b, LinOp* x) const override;
    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;
private:
    MyPrecond(const Factory*, std::shared_ptr<const LinOp>);
    MyPrecond(const Factory*, std::shared_ptr<const LinOp>, reuse_data_type&);
    static void check_reuse_consistent(const Factory*, const LinOp*,
                                       const reuse_data_type&);
};
```

The class provides three things on top of the plain factory:

- A `reuse_data_type` derived from `LinOpFactory::ReuseData`, which must
  be default-constructible. It records the setup results (aggregates,
  patterns, index maps) that stay valid while the pattern is unchanged.
- The constructor `MyPrecond(const Factory*, std::shared_ptr<const LinOp>,
  reuse_data_type&)`. The first call on an empty `reuse_data_type` is a
  full generation that fills it; later calls take the stored data and
  recompute only what depends on the values.
- The static function `check_reuse_consistent(const Factory*, const
  LinOp*, const reuse_data_type&)`, which throws when the input does not
  fit the stored data. It accepts any input while the data is empty.

Users then call `factory->create_empty_reuse_data()` once and pass it
to `factory->generate_reuse(A, *reuse_data)` for each matrix. See
[`gko::LinOpFactory`](../../../api-reference/base/lin-op-factory.md) for
the contract of the generated operator.

## The distributed preconditioner skeleton

The distributed classes live in
`namespace gko::experimental::distributed::preconditioner`.

```cpp
template <typename ValueType = default_precision,
          typename LocalIndexType = int32,
          typename GlobalIndexType = int64>
class MyPrecond
    : public LinOp {
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
        : LinOp(std::move(exec)) {}

    explicit MyPrecond(const Factory* factory,
                       std::shared_ptr<const LinOp> system_matrix)
        : LinOp(factory->get_executor(),
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
  `gko::experimental::distributed::preconditioner`, derived from `LinOp`
  without further bases. The header is wrapped
  in `#if GINKGO_BUILD_MPI` and the source is added to `core/CMakeLists.txt`
  under the `if(GINKGO_BUILD_MPI)` block.
- The static-assert is `GKO_ASSERT_SUPPORTED_VALUE_AND_DIST_INDEX_TYPE`,
  which constrains `LocalIndexType` and `GlobalIndexType` to the
  distributed-supported combinations.

The `apply_impl` for a purely local Schwarz-style preconditioner applies
the inner solver to the rank-local Dense slice and returns. The
Schwarz preconditioner does no MPI communication of its own:

```cpp
template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void MyPrecond<ValueType, LocalIndexType, GlobalIndexType>::apply_impl(
    const LinOp* b, LinOp* x) const
{
    experimental::precision_dispatch_real_complex_distributed<ValueType>(
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

`gko::detail::get_local(v)` is the internal helper in
`core/distributed/helpers.hpp` that Ginkgo's own solvers and
preconditioners use. It returns the rank-local
`matrix::Dense<V>*` for both `experimental::distributed::Vector` and ordinary
`matrix::Dense`, so the same iteration body covers serial and MPI use.

### Where MPI can live

A distributed preconditioner *can* do its own MPI communication, but
not everywhere:

- **Allowed**: core layer —
  `core/distributed/preconditioner/my_precond.cpp` and any host-side
  helpers it calls.
- **Forbidden**: kernel modules — `reference/`, `cuda/`, `hip/`,
  `omp/`, `dpcpp/`, `common/unified/`, `common/cuda_hip/` must not
  include `mpi.h` or touch the communicator.

The split is the same circular-dependency rule that applies to every
kernel module: backend kernels depend on `core/`, not the other way
around. The `no-circular-deps` CI job catches violations.

When communication is needed, structure `apply_dense_impl` as an
interleaving of kernel launches and MPI calls so MPI latency hides
behind on-device work:

1. Launch the kernel work that has no remote dependency.
2. Issue the non-blocking MPI call (`i_all_to_all_v`, `Isend`/`Irecv`,
   ...) from the host — returns immediately, kernel keeps running.
3. Wait on the MPI request, or use a stream event (see [Stream-MPI
   ordering for GPU executors](../../concepts/distributed/communication-primitives.md#stream-mpi-ordering-for-gpu-executors)).
4. Launch the kernel that consumes the received data.

The distributed `Matrix::apply` is the canonical example: diagonal
matrix kernel runs concurrently with the halo `i_all_to_all_v`, then
the off-diagonal kernel fires once the receive completes (see
[Communication
primitives](../../concepts/distributed/communication-primitives.md)).
The overlap holds on device executors for both communicators, because
the diagonal kernel is queued before the exchange is finalised. On host
executors it holds whenever the exchange is non-blocking: always for
`NeighborhoodCommunicator` (available except on Open MPI older than
4.1), and for the default `DenseCommunicator`
except on Open MPI older than 4.1, where the dense exchange falls back
to the blocking `MPI_Alltoallv` (see [Dense
communicator](../../concepts/distributed/dense-communicator.md)).
Replicate that structure rather than putting an `MPI_Allreduce`
between two synchronous kernel launches.

## Config / JSON registration

Add an entry to the `LinOpFactoryType` enum in
`core/config/config_helper.hpp` and a map entry to
`generate_config_map()` in `core/config/registry.cpp` (MPI-gate the
latter for a distributed preconditioner — see the Schwarz line as the
precedent). Then provide the parse specialisation:

- **Local, simple dispatch** — one line in
  `core/config/preconditioner_config.cpp` next to the Jacobi entry,
  using `GKO_PARSE_VALUE_AND_INDEX_TYPE` (or `GKO_PARSE_VALUE_TYPE`
  for a value-only preconditioner).
- **Local, custom dispatch** — its own file, mirroring
  `preconditioner_ilu_config.cpp` or `preconditioner_isai_config.cpp`.
- **Distributed** — always its own file under the
  `if(GINKGO_BUILD_MPI)` block of `core/CMakeLists.txt`. Copy
  `core/config/schwarz_config.cpp`; the hand-written
  `parse<LinOpFactoryType::Schwarz>` handles the `(LocalIndexType,
  GlobalIndexType)` dispatch and skips invalid combinations like
  `(int64, int32)`.

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

The full test plan is covered in [Write tests](write-tests.md). For a
new **local preconditioner** you write three files:

- `core/test/preconditioner/my_precond.cpp` — pure-C++ tests on the
  factory: `parameters_type` defaults, `with_*` setters, `build()`
  accepting a system matrix, `transpose()` returning a valid object.
  These don't need a backend or apply. Registered with
  `ginkgo_create_test(my_precond)` in
  `core/test/preconditioner/CMakeLists.txt`.
- `reference/test/preconditioner/my_precond_kernels.cpp` —
  Reference-only correctness tests on small hand-checkable inputs.
  Registered with `ginkgo_create_test(my_precond_kernels)`.
- `test/preconditioner/my_precond_kernels.cpp` — the **cross-backend
  parity test**. Same source file compiled once per enabled backend
  (`omp`, `cuda`, `hip`, `dpcpp`) by `ginkgo_create_common_test`,
  using `CommonTestFixture` (from `test/utils/common_fixture.hpp`)
  which provides both `ref` (Reference) and `exec` (the target
  backend). Each test sets up the same input on both, applies the
  preconditioner, and asserts agreement with
  `GKO_ASSERT_MTX_NEAR(...)`. Registered with
  `ginkgo_create_common_test(my_precond_kernels)` in
  `test/preconditioner/CMakeLists.txt`.

`jacobi*.cpp` in those three trees is the template to copy.

For a new **distributed preconditioner**, the cross-backend parity
test lives at `test/mpi/preconditioner/my_precond.cpp` and is
registered with `ginkgo_create_common_and_reference_test(my_precond
MPI_SIZE 3 LABELS distributed)`. That helper stamps out one test per
enabled backend *plus* a Reference variant, all run under MPI on the
requested rank count. `test/mpi/preconditioner/schwarz.cpp` is the
template.

:::{seealso}
- [Add a new solver](add-a-solver.md) — the iterative pattern, with
  workspace machinery.
- [Distributed solvers and preconditioners](../../concepts/distributed/solvers-and-preconditioners.md) — what the distributed
  preconditioner composes with.
- [Configure via JSON](../using/configure-via-json.md) — how config edits
  surface in user-facing JSON.
:::
