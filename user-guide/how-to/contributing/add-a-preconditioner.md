# Add a new preconditioner

A preconditioner in Ginkgo is a `LinOp` with two extra entry points:
`generate(system_matrix)` which builds the operator from a source matrix,
and `apply(b, x)` which applies the resulting operator. Preconditioners
do their work in a single `apply`, in one pass without an iteration
loop or workspace machinery.

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

```cpp
template <typename ValueType = default_precision, typename IndexType = int32>
class MyPrecond
    : public EnableLinOp<MyPrecond<ValueType, IndexType>>,
      public Transposable {
    friend class EnableLinOp<MyPrecond>;
    friend class EnablePolymorphicObject<MyPrecond, LinOp>;

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
                /* preconditioner state */ data_, dense_b, dense_x));
        },
        b, x);
}
```

The advanced overload passes `alpha` and `beta` through:

```cpp
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
                data_, dense_alpha, dense_b, dense_beta, dense_x));
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
- The supported `(LocalIndexType, GlobalIndexType)` combinations are the
  ones the instantiation block stamps out.
  `GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE` (see
  [Instantiation](#instantiation)) instantiates `(int32, int32)`,
  `(int32, int64)` and `(int64, int64)` for every value type.

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

`gko::detail::get_local(v)` is the internal helper (in
`core/distributed/helpers.hpp`) that Ginkgo's own solvers and
preconditioners use. It returns the rank-local `matrix::Dense<V>*` for
both `experimental::distributed::Vector` and ordinary `matrix::Dense`,
so the same body covers serial and MPI use.

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

1. Launch the kernel that packs the send buffer, then synchronize the
   executor so the host-side MPI call reads a complete buffer (see
   [Stream-MPI ordering for GPU
   executors](../../concepts/distributed/communication-primitives.md#stream-mpi-ordering-for-gpu-executors)).
2. Issue the non-blocking MPI call (`i_all_to_all_v`, `Isend`/`Irecv`,
   ...) from the host — with a non-blocking communicator it returns
   immediately (the blocking `DenseCommunicator` has completed the
   exchange by the time the call returns).
3. Launch the kernel work that has no remote dependency; with a
   non-blocking communicator it runs while the message is in flight.
4. Wait on the MPI request.
5. Launch the kernel that consumes the received data.

The distributed `Matrix::apply` is the canonical example: the local
block's kernel runs concurrently with the halo `i_all_to_all_v` when the
communicator is non-blocking (the default `NeighborhoodCommunicator`; with
`DenseCommunicator` the exchange has already completed when the local
kernel starts), then the non-local block's kernel fires once the receive
completes (see
[Communication
primitives](../../concepts/distributed/communication-primitives.md)).
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
