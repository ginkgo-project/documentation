# Add a new matrix format

A new sparse matrix format in Ginkgo is a `LinOp` subclass that owns its
storage arrays, plumbs `apply` into a backend-specific SpMV kernel, and
converts to/from Ginkgo's host-side assembly type (`matrix_data` /
`device_matrix_data`). This page walks through the skeleton, modelled on
the existing `matrix::Ell` (`include/ginkgo/core/matrix/ell.hpp`,
`core/matrix/ell.cpp`).

## What you commit to implement

A new matrix format provides at least:

| Trait | Why |
|---|---|
| `LinOp` base | Provides the public `apply` overloads; you write the two `apply_impl` overrides. |
| `EnableCloneable<Self>` | Provides `clone`, `copy_from`, `move_from` and `create_default`. It needs the copy and move assignment operators and a constructor that takes only the executor. See [Cloneable](../../../api-reference/base/cloneable.md). |
| `ReadableFromMatrixData<V, I>` | The `read` overloads that ingest `matrix_data` from host code. |
| `WritableToMatrixData<V, I>` | `write` to round-trip back to host-side `matrix_data`. |
| `ConvertibleTo<Dense<V>>`, `ConvertibleTo<Csr<V, I>>` | At minimum a path to and from CSR; Dense is conventional. |
| `ConvertibleTo<Self<next_precision<V>, I>>` | Cross-precision conversion to the neighbouring float type. |
| One backend kernel set | `spmv` plus `advanced_spmv` (alpha/beta variant); plus the `read` helpers (`compute_max_row_nnz`, `fill_in_matrix_data`) and a `convert_to_csr` kernel per backend. |

Conventional for current formats: `validate_data()` (structural and value
checks via `GKO_VALIDATE` and the helpers in `core/base/validation.hpp`;
throws `gko::InvalidData`; see
[`gko::PolymorphicObject`](../../../api-reference/base/polymorphic-object.md))
and the `get_device_view()` / `get_const_device_view()` pair that hands the
format's view structs to the kernels (see [Device views](#device-views)).

Optional but conventional: `DiagonalExtractable<V>`,
`EnableAbsoluteComputation<...>`, conversion to other formats (COO, ELL, ...).

## The class skeleton

Public header (`include/ginkgo/core/matrix/my_format.hpp`). The class
lives in `namespace gko::matrix`:

```cpp
template <typename ValueType = default_precision, typename IndexType = int32>
class MyFormat
    : public LinOp,
      public EnableCloneable<MyFormat<ValueType, IndexType>>,
      public ConvertibleTo<MyFormat<next_precision<ValueType>, IndexType>>,
      public ConvertibleTo<Dense<ValueType>>,
      public ConvertibleTo<Csr<ValueType, IndexType>>,
      public ReadableFromMatrixData<ValueType, IndexType>,
      public WritableToMatrixData<ValueType, IndexType>,
      public DiagonalExtractable<ValueType>,
      public EnableAbsoluteComputation<remove_complex<MyFormat<ValueType, IndexType>>> {
    friend class EnableCloneable<MyFormat>;
    GKO_ASSERT_SUPPORTED_VALUE_AND_INDEX_TYPE;

public:
    using EnableCloneable<MyFormat>::convert_to;
    using EnableCloneable<MyFormat>::move_to;
    // ... using-declarations for the other ConvertibleTo bases ...
    using value_type = ValueType;
    using index_type = IndexType;
    using mat_data        = matrix_data<ValueType, IndexType>;
    using device_mat_data = device_matrix_data<ValueType, IndexType>;
    using absolute_type   = remove_complex<MyFormat>;
    using device_view = matrix::view::my_format<value_type, index_type>;
    using const_device_view =
        matrix::view::my_format<const value_type, const index_type>;

    // Conversions (one pair per ConvertibleTo above):
    void convert_to(Dense<ValueType>* result) const override;
    void move_to   (Dense<ValueType>* result)       override;
    void convert_to(Csr<ValueType, IndexType>* result) const override;
    void move_to   (Csr<ValueType, IndexType>* result)       override;
    void convert_to(MyFormat<next_precision<ValueType>, IndexType>* result) const override;
    void move_to   (MyFormat<next_precision<ValueType>, IndexType>* result)       override;

    // Assembly ingress / egress:
    void read(const mat_data& data) override;
    void read(const device_mat_data& data) override;
    void read(device_mat_data&& data) override;
    void write(mat_data& data) const override;
    void validate_data() const override;

    // Auxiliary traits:
    std::unique_ptr<Diagonal<ValueType>> extract_diagonal() const override;
    std::unique_ptr<absolute_type>       compute_absolute() const override;
    void                                 compute_absolute_inplace() override;

    // Per-format storage accessors (data shape depends on your format):
    value_type*       get_values() noexcept { return values_.get_data(); }
    const value_type* get_const_values() const noexcept { return values_.get_const_data(); }
    index_type*       get_col_idxs() noexcept { return col_idxs_.get_data(); }
    const index_type* get_const_col_idxs() const noexcept { return col_idxs_.get_const_data(); }
    // ... dimensions, etc.

    device_view get_device_view();
    const_device_view get_const_device_view() const;

    static std::unique_ptr<MyFormat> create(
        std::shared_ptr<const Executor> exec, const dim<2>& size = {});

    // Required by EnableCloneable (clone, copy_from, move_from):
    MyFormat(const MyFormat&);
    MyFormat(MyFormat&&);
    MyFormat& operator=(const MyFormat&);
    MyFormat& operator=(MyFormat&&);

protected:
    // The executor-only constructor is what EnableCloneable calls.
    MyFormat(std::shared_ptr<const Executor> exec, const dim<2>& size = {});

    void resize(dim<2> new_size, size_type max_row_nnz);

    void apply_impl(const LinOp* b, LinOp* x) const override;
    void apply_impl(const LinOp* alpha, const LinOp* b,
                    const LinOp* beta, LinOp* x) const override;

private:
    array<value_type> values_;
    array<index_type> col_idxs_;
    // ... whatever else your storage scheme needs
};
```

A few notes on the base list:

- `LinOp` provides the public `apply` overloads; you only write
  `apply_impl`. `EnableCloneable<MyFormat>` provides cloning and
  same-type `copy_from` through the copy and move assignment operators,
  and creates its clones with the executor-only constructor, which is why
  the class befriends it.
- The half/bfloat16 precision strands (`next_precision<V, 2>`,
  `next_precision<V, 3>`) live under `#if GINKGO_ENABLE_HALF || ...`
  guards in `ell.hpp`. Copy that pattern if you want low-precision
  variants.
- `GKO_ASSERT_SUPPORTED_VALUE_AND_INDEX_TYPE` is a static-assert that
  catches accidentally-instantiated combinations outside the supported
  set (e.g. `(int8, int64)`).
- The constructors stay protected and each `create` overload is a
  `static` member you declare and define, for example
  `return std::unique_ptr<MyFormat>{new MyFormat{exec, size}};`. Users
  always go through `MyFormat::create(exec, ...)`, which returns a
  `std::unique_ptr<MyFormat>`.

### Device views

Each format with a view exposes two type aliases and a getter pair
(declared in the skeleton above), returning the non-owning
`matrix::view::my_format<...>` structs of
[`gko::matrix::view`](../../../api-reference/matrix/device-views.md):

```cpp
template <typename ValueType, typename IndexType>
typename MyFormat<ValueType, IndexType>::device_view
MyFormat<ValueType, IndexType>::get_device_view()
{
    return {this->get_size(), this->get_values(), this->get_col_idxs()};
}
```

Define `struct my_format` in `include/ginkgo/core/matrix/device_views.hpp`
next to `ell`. It is a plain aggregate of sizes, strides and raw pointers
with a `constexpr` constructor and an `as_const()` member, so it can be
passed by value into kernel code:

```cpp
namespace gko {
namespace matrix {
namespace view {

template <typename ValueType, typename IndexType>
struct my_format {
    dim<2> size;
    ValueType* values;
    IndexType* col_idxs;

    constexpr my_format(dim<2> size, ValueType* values, IndexType* col_idxs)
        : size{size}, values{values}, col_idxs{col_idxs}
    {}

    constexpr my_format<const ValueType, const IndexType> as_const() const
    {
        return {size, values, col_idxs};
    }
};

}  // namespace view
}  // namespace matrix
}  // namespace gko
```

`get_const_device_view()` has the same body as `get_device_view()` with
`get_const_values()` and `get_const_col_idxs()`.

### Validation

Override `validate_data()` so that users can check data they filled in
from outside. It throws `gko::InvalidData` when the stored arrays violate
the format's invariants, and runs when the user calls it. `Ell` and `Csr`
are the precedents:

```cpp
template <typename ValueType, typename IndexType>
void MyFormat<ValueType, IndexType>::validate_data() const
{
    GKO_VALIDATE(validation::is_within_nonegative_bounds(
                     col_idxs_, static_cast<IndexType>(this->get_size()[1])),
                 "col_idxs must be within [0, num_cols)");
    GKO_VALIDATE(validation::sparse_matrix_values_are_finite(values_),
                 "matrix must contain only finite values");
}
```

`GKO_VALIDATE` and the helpers in `gko::validation` (`is_sorted`,
`is_within_nonegative_bounds`, `has_unique_idxs_in_row`,
`sparse_matrix_values_are_finite`) come from `core/base/validation.hpp`.
Each helper copies its array to the host, so a call costs a
device-to-host transfer. Formats whose layout needs its own checks add
helper functions next to the format, as `core/matrix/ell.cpp` does.

## apply_impl

The `LinOp` base provides the public `apply` overloads; you implement
the two `apply_impl` overrides. Pick the dispatch helper by
the SpMV kernel's signature:

- **`mixed_precision_dispatch_real_complex<ValueType>(fn, b, x)`** if the
  kernel is templated on three independent value types
  (`InputValueType, MatrixValueType, OutputValueType`), as in CSR, ELL,
  SparsityCsr. Only the 3-arg form exists, so the advanced apply
  captures `alpha`/`beta` into the lambda.
- **`precision_dispatch_real_complex<ValueType>(fn, ...)`** if the kernel
  is templated on a single `ValueType`, as in COO, SELL-P, FBCSR, Diagonal,
  Hybrid, Dense, ScaledPermutation. Has both 3-arg and 5-arg forms.

The example below shows the mixed-precision pattern; for a
single-precision kernel, use the non-mixed helper and pass `alpha, b,
beta, x` directly into the 5-arg form.

```cpp
template <typename ValueType, typename IndexType>
void MyFormat<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    mixed_precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->get_executor()->run(my_format::make_spmv(
                this->get_const_device_view(),
                dense_b->get_const_device_view(),
                dense_x->get_device_view()));
        },
        b, x);
}

template <typename ValueType, typename IndexType>
void MyFormat<ValueType, IndexType>::apply_impl(const LinOp* alpha,
                                                const LinOp* b,
                                                const LinOp* beta,
                                                LinOp* x) const
{
    mixed_precision_dispatch_real_complex<ValueType>(
        [this, alpha, beta](auto dense_b, auto dense_x) {
            auto dense_alpha = make_temporary_conversion<ValueType>(alpha);
            auto dense_beta  = make_temporary_conversion<
                typename std::decay_t<decltype(*dense_x)>::value_type>(beta);
            this->get_executor()->run(my_format::make_advanced_spmv(
                dense_alpha->get_const_device_view(),
                this->get_const_device_view(),
                dense_b->get_const_device_view(),
                dense_beta->get_const_device_view(),
                dense_x->get_device_view()));
        },
        b, x);
}
```

Notes on the snippet:

- `make_temporary_conversion<ValueType>(alpha)` returns a
  `temporary_conversion<Dense<ValueType>>`: a unique-pointer-like
  wrapper that holds either the existing `Dense` (no copy) or a fresh
  conversion to `ValueType`. Access via `->` or `.get()`. Destroyed
  with the lambda, so the conversion is bounded by the apply.
- `my_format::make_spmv` / `make_advanced_spmv` are the operation
  builders produced by `GKO_REGISTER_OPERATION` (see [Add a new
  kernel](add-a-kernel.md)).
- All kernel arguments use views (`matrix::view::dense<...>`,
  `matrix::view::<your_format><...>`) rather than raw `Dense*` /
  `Format*`. Views make const-ness and value-type explicit in the
  signature and are the convention for new formats. `Csr`, `Coo`, `Ell`,
  `Sellp`, `Hybrid` and `Dense` all expose device views and pass them to
  their kernels. `Ell` and `Coo` are the smallest precedents to copy.

## read / write: host-side assembly

The mandatory `read(matrix_data)` ingests Ginkgo's host-side
incremental-assembly type. The conventional shape: count nonzeros to size
your arrays, then hand the device-side fill-in to a kernel:

```cpp
template <typename ValueType, typename IndexType>
void MyFormat<ValueType, IndexType>::read(const device_mat_data& data)
{
    auto exec = this->get_executor();
    auto local_data = make_temporary_clone(exec, &data);
    // Compute the layout (max nnz per row, padding, etc.) on the
    // executor, as Ell does, then resize the storage:
    array<int64> row_ptrs(exec, data.get_size()[0] + 1);
    exec->run(my_format::make_convert_idxs_to_ptrs(
        local_data->get_const_row_idxs(), local_data->get_num_stored_elements(),
        data.get_size()[0], row_ptrs.get_data()));
    size_type max_row_nnz{};
    exec->run(my_format::make_compute_max_row_nnz(row_ptrs, max_row_nnz));
    this->resize(data.get_size(), max_row_nnz);
    // Hand off to the backend kernel:
    exec->run(my_format::make_fill_in_matrix_data(
        *local_data, row_ptrs.get_const_data(), this->get_device_view()));
}

template <typename ValueType, typename IndexType>
void MyFormat<ValueType, IndexType>::read(const mat_data& data)
{
    this->read(device_mat_data::create_from_host(this->get_executor(), data));
}

template <typename ValueType, typename IndexType>
void MyFormat<ValueType, IndexType>::read(device_mat_data&& data)
{
    this->read(data);
    data.empty_out();
}
```

`my_format::make_convert_idxs_to_ptrs`, `make_compute_max_row_nnz` and
`make_fill_in_matrix_data` are operations you register in
`core/matrix/my_format.cpp` with `GKO_REGISTER_OPERATION`, as
`core/matrix/ell.cpp` does: the first from the shared
`components::convert_idxs_to_ptrs`, the other two from your format's
kernels.

`write` is the reverse:

```cpp
template <typename ValueType, typename IndexType>
void MyFormat<ValueType, IndexType>::write(mat_data& data) const
{
    // Walk the local storage on the host (use a temporary clone if your
    // matrix lives on the device) and append triplets to `data.nonzeros`.
}
```

## Conversions

`ConvertibleTo<...>` bases are not strictly required by anything in
`LinOp`, but the conventions across shipped formats are clear:

- **`Csr<V, I>`**: provide it. CSR is the canonical sparse interop
  target; every sparse format in the tree converts to/from CSR. Many
  algorithms and the benchmark suite rely on this path. Typically
  implemented as a kernel (`my_format::make_convert_to_csr`) because
  it's a parallel storage-layout transformation.
- **`Dense<V>`**: provide it unless your storage cannot be densified
  meaningfully (the only shipped exception is `Diagonal`, which omits
  `Dense`). The implementation can route through CSR or fill the dense
  buffer directly.
- **`MyFormat<next_precision<V>, I>`**: needed for cross-precision
  interop. The body is a member-wise assignment plus `set_size`. Add
  the `next_precision<V, 2>` / `next_precision<V, 3>` strands under
  `#if GINKGO_ENABLE_HALF || GINKGO_ENABLE_BFLOAT16` guards if you
  want half/bfloat16 support.

## Kernel files

Each operation `apply_impl` and `convert_to` calls needs the four files
described in [Add a new kernel](add-a-kernel.md). For a new format
`my_format`, the layout is:

| File | Purpose |
|---|---|
| `core/matrix/my_format_kernels.hpp` | Kernel macro declarations |
| `reference/matrix/my_format_kernels.cpp` | Reference single-thread impl |
| `omp/matrix/my_format_kernels.cpp` | OMP impl (if hand-tuned) |
| `common/unified/matrix/my_format_kernels.cpp` | Portable CUDA/HIP/SYCL/OMP impl |
| `common/cuda_hip/matrix/my_format_kernels.cpp` (optional) | Hand-tuned CUDA / HIP, one source for both |
| `dpcpp/matrix/my_format_kernels.dp.cpp` (optional) | Hand-tuned SYCL |

## CMake wiring

A new format **does** require CMake edits, unlike a new kernel on an
existing format. Add the file to each backend's source list:

- `core/CMakeLists.txt`: add `matrix/my_format.cpp` next to
  `matrix/ell.cpp`.
- `reference/CMakeLists.txt`, `omp/CMakeLists.txt`,
  `dpcpp/CMakeLists.txt`: add `matrix/my_format_kernels.cpp`
  (`.dp.cpp` for DPC++) next to `matrix/ell_kernels.cpp`.
- `common/unified/CMakeLists.txt`: add
  `matrix/my_format_kernels.cpp` if you have a portable kernel.
- `common/cuda_hip/CMakeLists.txt`: add
  `matrix/my_format_kernels.cpp` only if you wrote a hand-tuned
  CUDA / HIP variant.

The public header `include/ginkgo/core/matrix/my_format.hpp` becomes
discoverable through the standard install layout once the file exists in
the tree, with no CMake list to update for the include.

## Config layer and benchmark suite

- The runtime JSON / `gko::config` registry is for *factories*
  (solvers, preconditioners, factorizations).
- Matrix formats are constructed directly:
  `MyFormat::create(exec, ...)`.
- To make the format selectable through the benchmark suite's
  `FORMATS=...` selector, extend `benchmark/utils/formats.hpp` (maps
  format names to matrix factories).

## Reference-parity test

The full test plan is covered in [Write tests](write-tests.md). For a
new format you write two files:

- `reference/test/matrix/my_format_kernels.cpp`: small,
  hand-checkable inputs run only on the Reference executor. Covers
  the SpMV kernels, the conversions, and the `read`/`write`
  round-trip. Register with `ginkgo_create_test(my_format_kernels)`
  in `reference/test/matrix/CMakeLists.txt`.
- `test/matrix/my_format_kernels.cpp`: the **cross-backend parity
  tests**. Same source file compiled once per enabled backend
  (`omp`, `cuda`, `hip`, `dpcpp`) by `ginkgo_create_common_test`,
  using `CommonTestFixture` (from `test/utils/common_fixture.hpp`)
  which provides both `ref` (Reference) and `exec` (the target
  backend). Each test runs the same input on both executors and
  asserts agreement with `GKO_ASSERT_MTX_NEAR(...)`. Register with
  `ginkgo_create_common_test(my_format_kernels)` in
  `test/matrix/CMakeLists.txt`. Use inputs at least the warp size
  (typically 100×100 or larger) so warp-boundary corner cases are
  exercised.

The per-backend test trees (for example `hip/test/matrix/`) hold only
the few tests that need a backend-specific path. Put new tests in
`test/matrix/`.

:::{seealso}
- [Add a new kernel](add-a-kernel.md): the kernel files that back `apply_impl`.
- [Write tests](write-tests.md): the reference-parity pattern.
- [Run the benchmark suite](run-benchmark-suite.md): extend the SpMV
  benchmark to include your new format.
:::
