# Views and zero-copy wrapping

An `array` can wrap memory you already own — a buffer from deal.II, MFEM, Kokkos, or a raw user pointer — without copying it. This is essential when integrating Ginkgo as a solver backend inside a larger simulation code: you can hand Ginkgo a view of your existing data, run a solve, and read the result back in place, all without a single extra allocation. This page explains the lifetime contract, the differences between owned and viewed arrays, and the patterns for the most common integration scenarios.

## Constructing a view

Use `gko::make_array_view` to create a non-owning `array`:

```cpp
double*           user_data = ...;   // memory the user already owns, accessible by exec
gko::size_type    n         = ...;
auto view = gko::make_array_view(exec, n, user_data);
```

`view` is a `gko::array<double>` that does not own `user_data`. When `view` goes out of scope, it does not free the memory. The size `n` must be at most the number of elements allocated at `user_data`.

For read-only access, use `gko::make_const_array_view`:

```cpp
const double* const_data = ...;
auto const_view = gko::make_const_array_view(exec, n, const_data);
```

The type returned by `make_const_array_view` is an implementation detail. Assign it to `auto`. It can be passed to Ginkgo functions that accept const array-like objects (typically `create_const` factory methods on matrix types).

A view can also be created from an existing array:

```cpp
gko::array<double> a(exec, 10);
auto view  = a.as_view();         // non-owning, mutable
auto cview = a.as_const_view();   // non-owning, read-only
```

The owning status of any array is queryable:

```cpp
auto owning     = gko::array<double>(exec, 5);
auto non_owning = gko::make_array_view(exec, 5, data);
assert(owning.is_owning()     == true);
assert(non_owning.is_owning() == false);
```

## The lifetime contract

:::{attention}
The viewed memory must outlive the view. Ginkgo does not track ownership of viewed buffers. If `user_data` is freed before `view` is destroyed, any subsequent access to `view` is undefined behaviour.

Treat views as *borrows*. The safest pattern is to create the view, use it for a single operation (such as one solver call), and let it go out of scope before the underlying buffer can be released.
:::

The executor compatibility rule also applies: `user_data` must be accessible by `exec`. A view backed by host memory cannot be passed to a `CudaExecutor`, and vice versa. Mixing an incompatible pointer and executor produces undefined behaviour at runtime.

## Owned vs viewed semantics

| Operation                   | Owned `array`                               | Viewed `array`                                 |
| --------------------------- | ------------------------------------------- | ---------------------------------------------- |
| Destructor                  | Frees the memory                            | Does nothing to the memory                     |
| Copy (`array b = a;`)       | Allocates new memory and copies             | Allocates new memory and copies                |
| Move                        | Transfers ownership                         | Transfers the view (still non-owning)          |
| `clone(exec, a)`            | Cross-executor copy (new allocation)        | Cross-executor copy (new allocation)           |
| `resize_and_reset(n)`       | Reallocates to new size                     | **Not supported** — throws unless `n` matches the current size (a no-op) |
| `fill(value)`               | Sets all elements                           | Sets all elements (writes through to buffer)   |

:::{note}
Resizing a view is not supported — Ginkgo does not own the underlying buffer and cannot grow it. If your computation needs more space than the user provided, copy into a new owned array with `gko::clone` or the cross-executor constructor.
:::

## Integration patterns

### deal.II distributed vector

deal.II's `LinearAlgebra::distributed::Vector` exposes a raw pointer to the locally-owned portion of its data. Wrapping it in a view lets Ginkgo operate on the local block without any copy:

```cpp
dealii::LinearAlgebra::distributed::Vector<double> dealii_vec(...);
auto exec = gko::OmpExecutor::create();

// Wrap the local data as a non-owning array.
auto local_view = gko::make_array_view(exec,
                                        dealii_vec.local_size(),
                                        dealii_vec.get_values());

// Build a Dense multi-vector around the view for solver input/output.
auto dense = gko::matrix::Dense<double>::create(
                 exec,
                 gko::dim<2>{dealii_vec.local_size(), 1},
                 std::move(local_view),
                 /*stride=*/1);
```

`dealii_vec` must outlive `dense` (and transitively, the solver call that uses it).

### Raw user buffer

When you have a raw pointer to host memory and want Ginkgo to operate on it directly:

```cpp
double*        buf      = /* allocated by the user */;
gko::size_type buf_size = ...;

auto host = gko::ReferenceExecutor::create();
auto view = gko::make_array_view(host, buf_size, buf);

// Build any Ginkgo type around the view.
// For example, a Dense vector:
auto vec = gko::matrix::Dense<double>::create(
               host, gko::dim<2>{buf_size, 1},
               std::move(view), /*stride=*/1);
```

### Kokkos View

A Kokkos `View` can be wrapped the same way, provided the Kokkos memory space matches the Ginkgo executor's memory space. For a `Kokkos::View<double*>` in `HostSpace`, use a `ReferenceExecutor` or `OmpExecutor`. For `CudaSpace`, use a `CudaExecutor`. Extract the raw data pointer with `kokkos_view.data()` and pass it to `make_array_view`.

```cpp
Kokkos::View<double*, Kokkos::HostSpace> kv("buf", n);
auto exec = gko::OmpExecutor::create();
auto view = gko::make_array_view(exec, n, kv.data());
```

As with all views, the Kokkos view must outlive the Ginkgo array view.

### Constructing a CSR matrix from non-owned views

Most matrix formats — including CSR — can be constructed directly from `array<T>` arguments. The relevant `create` overload takes the arrays *by value*:

```cpp
static std::unique_ptr<Csr> create(
    std::shared_ptr<const Executor> exec, const dim<2>& size,
    array<value_type> values, array<index_type> col_idxs,
    array<index_type> row_ptrs,
    std::shared_ptr<strategy_type> strategy = nullptr);
```

If you pass **rvalue** arrays (via `std::move` or freshly-constructed temporaries), Ginkgo moves them into the matrix directly — no copy. When the moved-in array is a view, the resulting matrix simply references the same user memory:

```cpp
// User owns these buffers — Ginkgo will not free them.
double* values    = ...;        // length nnz
int*    col_idxs  = ...;        // length nnz
int*    row_ptrs  = ...;        // length n_rows + 1

auto exec = gko::OmpExecutor::create();

auto mat = gko::matrix::Csr<double, int>::create(
               exec,
               gko::dim<2>{n_rows, n_cols},
               gko::make_array_view(exec, nnz,         values),    // rvalue view
               gko::make_array_view(exec, nnz,         col_idxs),  // rvalue view
               gko::make_array_view(exec, n_rows + 1,  row_ptrs)); // rvalue view
```

The returned `mat` is a CSR matrix that operates on the user's existing buffers. SpMV reads from them; in-place operations write to them.

:::{attention}
The CSR `create` API has an explicit caveat in its documentation: *"If one of `row_ptrs`, `col_idxs` or `values` is not an rvalue, not an array of `IndexType`, `IndexType` and `ValueType` respectively, or is on the wrong executor, an internal copy of that array will be created."* In other words: if you forget to `std::move` an lvalue array, or the executor doesn't match, Ginkgo silently copies. The matrix is correct, but you've lost the zero-copy property. Always pass rvalues when zero-copy matters.
:::

For read-only buffers, use `Csr::create_const`, which takes `const_array_view` rvalues and produces a `unique_ptr<const Csr>`:

```cpp
auto mat = gko::matrix::Csr<double, int>::create_const(
               exec,
               gko::dim<2>{n_rows, n_cols},
               gko::make_const_array_view(exec, nnz,        const_values),
               gko::make_const_array_view(exec, nnz,        const_col_idxs),
               gko::make_const_array_view(exec, n_rows + 1, const_row_ptrs));
```

The resulting matrix can be applied (`mat->apply(b, x)`), used as a system matrix in a solver factory, etc., but cannot be modified — enforced at the type level.

The same pattern works for `Coo`, `Ell`, `Dense`, and the other matrix formats: each provides a `create(exec, size, ...arrays)` overload that moves rvalue arrays in directly, plus a `create_const(...)` for immutable wrapping.

## When to copy instead of view

Views eliminate allocation overhead, but they are not always the right choice:

- **Short-lived operations** — if the view lives for a single `apply` call and the source buffer is known to be valid throughout, a view is safe and efficient.
- **Cross-executor mismatch** — if the source buffer is on the host but the computation runs on a GPU, a view cannot bridge that gap. Copy explicitly with `gko::clone(gpu_exec, host_array)`.
- **Uncertain buffer lifetime** — if you cannot guarantee that the source buffer outlives all uses of the view, copy into an owned array first.
- **Need to resize** — views cannot be resized. If the algorithm needs a scratch buffer of unknown size, allocate an owned array.

:::{seealso}
- [Memory ownership and `gko::array`](memory-ownership.md) — owned-array semantics.
- [The Executor model](executor-model.md) — memory space compatibility.
- API reference: [`gko::make_array_view`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/namespacegko.html)
:::
