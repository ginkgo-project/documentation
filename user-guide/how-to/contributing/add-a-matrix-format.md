# Add a new matrix format

A new sparse matrix format in Ginkgo is a `LinOp` subclass that owns its
storage arrays, plumbs `apply` into a backend-specific SpMV kernel, and
converts to/from Ginkgo's host-side assembly type (`matrix_data` /
`device_matrix_data`). This page walks through the skeleton, modelled on
the existing `matrix::Ell` (`include/ginkgo/core/matrix/ell.hpp`,
`core/matrix/ell.cpp`).

## What you commit to implement

Every matrix format provides at least:

| Trait | Why |
|---|---|
| `EnableLinOp<Self>` base | Auto-generates the `apply` dispatcher; you only write `apply_impl`. |
| `ReadableFromMatrixData<V, I>` | The `read` overloads that ingest `matrix_data` from host code. |
| `WritableToMatrixData<V, I>` | `write` to round-trip back to host-side `matrix_data`. |
| `ConvertibleTo<Dense<V>>`, `ConvertibleTo<Csr<V, I>>` | At minimum a path to and from CSR; Dense is conventional. |
| `ConvertibleTo<Self<next_precision<V>, I>>` | Cross-precision conversion to the neighbouring float type. |
| One backend kernel set | `spmv` plus `advanced_spmv` (alpha/beta variant); plus a `read` / `convert_to_csr` kernel per backend. |

Optional but conventional: `DiagonalExtractable<V>`,
`EnableAbsoluteComputation<...>`, conversion to other formats (COO, ELL, ...).

## The class skeleton

Public header (`include/ginkgo/core/matrix/my_format.hpp`):

```cpp
template <typename ValueType = default_precision, typename IndexType = int32>
class MyFormat
    : public EnableLinOp<MyFormat<ValueType, IndexType>>,
      public ConvertibleTo<MyFormat<next_precision<ValueType>, IndexType>>,
      public ConvertibleTo<Dense<ValueType>>,
      public ConvertibleTo<Csr<ValueType, IndexType>>,
      public ReadableFromMatrixData<ValueType, IndexType>,
      public WritableToMatrixData<ValueType, IndexType>,
      public DiagonalExtractable<ValueType>,
      public EnableAbsoluteComputation<remove_complex<MyFormat<ValueType, IndexType>>> {
    friend class EnablePolymorphicObject<MyFormat, LinOp>;

public:
    using value_type = ValueType;
    using index_type = IndexType;
    using mat_data        = matrix_data<ValueType, IndexType>;
    using device_mat_data = device_matrix_data<ValueType, IndexType>;
    using absolute_type   = remove_complex<MyFormat>;

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

    // Auxiliary traits:
    std::unique_ptr<Diagonal<ValueType>> extract_diagonal() const override;
    std::unique_ptr<absolute_type>       compute_absolute() const override;
    void                                 compute_absolute_inplace() override;

    static std::unique_ptr<MyFormat> create(
        std::shared_ptr<const Executor> exec, const dim<2>& size = {});

    // Per-format storage accessors (data shape depends on your format):
    value_type*       get_values() noexcept { return values_.get_data(); }
    const value_type* get_const_values() const noexcept { return values_.get_const_data(); }
    // ... index arrays, dimensions, etc.

protected:
    // Constructors live in the protected section; users go through ::create.
    MyFormat(std::shared_ptr<const Executor> exec, const dim<2>& size = {});

    void apply_impl(const LinOp* b, LinOp* x) const override;
    void apply_impl(const LinOp* alpha, const LinOp* b,
                    const LinOp* beta, LinOp* x) const override;

    // Re-allocates the storage for a new size and per-row shape:
    void resize(dim<2> new_size, size_type max_row_nnz);

private:
    array<value_type> values_;
    array<index_type> col_idxs_;
    // ... whatever else your storage scheme needs
};
```

A few notes on the base list:

- `EnableLinOp<MyFormat>` injects the public `apply` overloads. You
  only write `apply_impl`, and you declare the static `create(...)`
  yourself, as `Ell` does in `ell.hpp`.
- The half/bfloat16 precision strands (`next_precision<V, 2>`,
  `next_precision<V, 3>`) live under `#if GINKGO_ENABLE_HALF || ...`
  guards in `ell.hpp`. Copy that pattern if you want low-precision
  variants.
- The supported `(ValueType, IndexType)` combinations are the ones
  `core/matrix/my_format.cpp` instantiates. End that file the way
  `ell.cpp` does: define `GKO_DECLARE_MY_FORMAT_MATRIX(ValueType,
  IndexType)` as `class MyFormat<ValueType, IndexType>` and pass it to
  `GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE`.
- The protected constructor plus a static `create` is deliberate.
  Users always go through `MyFormat::create(exec, ...)`, which returns
  a `std::unique_ptr<MyFormat>`.

## apply_impl

The `EnableLinOp` base auto-generates the public `apply` overloads; you
implement the two `apply_impl` overrides. Pick the dispatch helper by
the SpMV kernel's signature:

- **`mixed_precision_dispatch_real_complex<ValueType>(fn, b, x)`** if the
  kernel is templated on three independent value types
  (`InputValueType, MatrixValueType, OutputValueType`) — CSR, ELL,
  SparsityCsr. Only the 3-arg form exists, so the advanced apply
  captures `alpha`/`beta` into the lambda.
- **`precision_dispatch_real_complex<ValueType>(fn, ...)`** if the kernel
  is templated on a single `ValueType` — COO, SELL-P, FBCSR, Diagonal,
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
            this->get_executor()->run(
                my_format::make_spmv(this, dense_b, dense_x));
        },
        b, x);
}
```

The advanced overload additionally converts `alpha` and `beta`:

```cpp
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
                dense_alpha.get(), this, dense_b, dense_beta.get(), dense_x));
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
- Kernel arguments are the matrix objects themselves:
  `const matrix::MyFormat<ValueType, IndexType>*` for inputs,
  `matrix::MyFormat<...>*` for outputs, and
  `const matrix::Dense<...>*` / `matrix::Dense<...>*` for the dense
  operands, as in the `Ell` and `Csr` kernel headers. Inside the
  kernel, read raw arrays through `get_const_values()`,
  `get_const_col_idxs()` and the other accessors.

## read / write — host-side assembly

The mandatory `read(matrix_data)` ingests Ginkgo's host-side
incremental-assembly type. The conventional shape: count nonzeros to size
your arrays, then hand the device-side fill-in to a kernel:

```cpp
template <typename ValueType, typename IndexType>
void MyFormat<ValueType, IndexType>::read(const device_mat_data& data)
{
    auto exec = this->get_executor();
    auto local_data = make_temporary_clone(exec, &data);
    // Compute the layout (here: max nnz per row) from `data` on the executor.
    array<int64> row_ptrs(exec, data.get_size()[0] + 1);
    exec->run(my_format::make_convert_idxs_to_ptrs(
        local_data->get_const_row_idxs(), local_data->get_num_stored_elements(),
        data.get_size()[0], row_ptrs.get_data()));
    size_type max_row_nnz{};
    exec->run(my_format::make_compute_max_row_nnz(row_ptrs, max_row_nnz));
    // Resize storage, then hand the fill-in to the backend kernel:
    this->resize(data.get_size(), max_row_nnz);
    exec->run(my_format::make_fill_in_matrix_data(
        *local_data, row_ptrs.get_const_data(), this));
}
```

The other two overloads forward to it:

```cpp
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

- **`Csr<V, I>`** — provide it. CSR is the canonical sparse interop
  target; every sparse format in the tree converts to/from CSR. Many
  algorithms and the benchmark suite rely on this path. Typically
  implemented as a kernel (`my_format::make_convert_to_csr`) because
  it's a parallel storage-layout transformation.
- **`Dense<V>`** — provide it unless your storage cannot be densified
  meaningfully (the only shipped exception is `Diagonal`, which omits
  `Dense`). The implementation can route through CSR or fill the dense
  buffer directly.
- **`MyFormat<next_precision<V>, I>`** — needed for cross-precision
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
| `common/cuda_hip/matrix/my_format_kernels.cpp` (optional) | Hand-tuned CUDA and HIP, compiled once for both |
| `dpcpp/matrix/my_format_kernels.dp.cpp` (optional) | Hand-tuned SYCL |

## CMake wiring

A new format **does** require CMake edits, unlike a new kernel on an
existing format. Add the file to each backend's source list:

- `core/CMakeLists.txt` — add `matrix/my_format.cpp` next to
  `matrix/ell.cpp`.
- `reference/CMakeLists.txt`, `omp/CMakeLists.txt`,
  `dpcpp/CMakeLists.txt` — add `matrix/my_format_kernels.cpp`
  (`.dp.cpp` for DPC++) next to `matrix/ell_kernels.cpp`.
- `common/unified/CMakeLists.txt` — add
  `matrix/my_format_kernels.cpp` if you have a portable kernel.
- `common/cuda_hip/CMakeLists.txt` — add
  `matrix/my_format_kernels.cpp` only if you wrote a hand-tuned CUDA/HIP
  variant.

The public header `include/ginkgo/core/matrix/my_format.hpp` becomes
discoverable through the standard install layout once the file exists in
the tree — no CMake list to update for the include.

## Matrix formats are constructed directly

- The runtime JSON / `gko::config` registry is for *factories*
  (solvers, preconditioners, factorizations).
- Matrix formats are constructed directly:
  `MyFormat::create(exec, ...)`.
- If you want the format to participate in the benchmark suite's
  `FORMATS=...` selector, add an entry to the format map in
  `benchmark/utils/formats.hpp`; `benchmark/utils/types.hpp` only holds
  the `etype` / `itype` aliases.

## Reference-parity test

The full test plan is covered in [Write tests](write-tests.md). For a
new format you write two files:

- `reference/test/matrix/my_format_kernels.cpp` — small,
  hand-checkable inputs run only on the Reference executor. Covers
  the SpMV kernels, the conversions, and the `read`/`write`
  round-trip. Register with `ginkgo_create_test(my_format_kernels)`
  in `reference/test/matrix/CMakeLists.txt`.
- `test/matrix/my_format_kernels.cpp` — the **cross-backend parity
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

The per-backend test trees (for example `hip/test/matrix/`) hold only the
few tests that need a backend-specific path. Put new tests in
`test/matrix/`.

:::{seealso}
- [Add a new kernel](add-a-kernel.md) — the kernel files that back `apply_impl`.
- [Write tests](write-tests.md) — the reference-parity pattern.
- [Run the benchmark suite](run-benchmark-suite.md) — extend the SpMV
  benchmark to include your new format.
:::
