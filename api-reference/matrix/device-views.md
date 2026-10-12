# `gko::matrix::view`

Each of `Dense`, `Coo`, `Csr`, `Ell`, `Sellp` and `Hybrid` exposes `device_view` and `const_device_view` types, together with the member functions `get_device_view()` and `get_const_device_view()`. A view is a small, trivially copyable struct of sizes and raw pointers, meant to be passed by value into kernel code or across library boundaries. The pointers refer to memory on the matrix's executor and stay valid as long as the matrix is alive and not resized.

```cpp
// A read-only view of A's arrays, e.g. to pass into a kernel.
auto view = A->get_const_device_view();
auto num_rows = view.size[0];
auto num_entries = view.num_stored_elements;
const int* row_ptrs = view.row_ptrs;  // col_idxs and values likewise
```

The view types live in `<ginkgo/core/matrix/device_views.hpp>`. `get_device_view()` returns a view with mutable pointers; `as_const()` converts it to its const counterpart.

## Views

```{doxygenstruct} gko::matrix::view::dense
:members:
```

```{doxygenstruct} gko::matrix::view::coo
:members:
```

```{doxygenstruct} gko::matrix::view::csr
:members:
```

```{doxygenstruct} gko::matrix::view::ell
:members:
```

```{doxygenstruct} gko::matrix::view::sellp
:members:
```

```{doxygenstruct} gko::matrix::view::hybrid
:members:
```

:::{seealso}
- [Matrix formats](../../user-guide/concepts/matrix-formats.md) — the storage each view describes.
- [`gko::matrix`](index.md) — the matrix classes that hand out views.
:::
