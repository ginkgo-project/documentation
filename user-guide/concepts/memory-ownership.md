# Memory ownership and gko::array

`gko::array<T>` is the basic owning container for contiguous data in Ginkgo. Every higher-level container (vectors, matrices, solver workspaces) is built on it. This page explains how `array` owns memory, what its copy and move semantics are, and how it interacts with executors. View semantics (wrapping memory you do not own) are covered separately in [Views and zero-copy wrapping](views-zero-copy.md).

## What gko::array stores

`array<T>` is templated by the storage type. The type must be a POD type (plain old data): a built-in type, or a struct of built-in types. Examples include `double`, `float`, `int`, `std::complex<double>`. The type **may not** be const-qualified; use a `const array<T>` to express read-only access to the whole container, rather than `array<const T>`.

:::{attention}
Some functionality (notably `fill`) is not available for all storage types. The restriction is enforced at link time. If you use an unsupported type, the error message will look like:

```
undefined reference to gko::array<T>::fill(T)
```

The supported types are documented in the Resizing and filling section below.
:::

## Construction

An array is constructed from an executor and the number of stored elements:

```cpp
std::shared_ptr<const gko::Executor> exec = gko::ReferenceExecutor::create();
gko::size_type size = 10;
gko::array<double> a(exec, size);
```

:::{attention}
After construction with a size only, the stored memory is *not* initialized. Reading from `a` before writing to it is undefined behaviour.
:::

To create an array that is already filled with data, pass an initializer list instead of a size. The initializer list lives in CPU memory, but the created array will place the elements into the memory space of the executor:

```cpp
gko::array<double> b(exec, {1.0, 2.0, 3.0});
```

An alternative is to provide begin and end iterators from any host container. The data is copied into the array:

```cpp
std::vector<double> std_c{1.0, 2.0, 3.0};
gko::array<double> c(exec, std_c.begin(), std_c.end());
```

All four common construction forms together:

```cpp
// Size only — uninitialised
gko::array<double> a(exec, size);

// Initializer list
gko::array<double> b(exec, {1.0, 2.0, 3.0});

// Iterator pair from host container
std::vector<double> std_c{1.0, 2.0, 3.0};
gko::array<double> c(exec, std_c.begin(), std_c.end());

// Default construction
gko::array<double> d;        // no executor, zero size
gko::array<double> e(exec);  // executor, zero size
```

The default-constructed array `d` is not associated with any executor. The empty array `e` is associated with `exec` but stores nothing. The difference matters for copy and move operations.

## Accessing the data

The primary access point is the pair `get_data()` / `get_const_data()`:

```cpp
double*       ptr       = a.get_data();        // non-const only
const double* const_ptr = a.get_const_data();  // always available
```

`get_data()` is unavailable on `const` arrays. Both return `nullptr` if the array is empty.

:::{note}
The pattern of having `get_<something>` and `get_const_<something>` (mutable and const accessors) is standard throughout Ginkgo. For `const` objects, only the const accessor is callable, making accidental mutation near-impossible.
:::

:::{attention}
The returned pointers address the executor's memory space. If the array lives on a `CudaExecutor`, the pointer is a device pointer that is not dereferenceable on the host. Dereference the pointer only on the device that owns it.
:::

Element count and executor:

```cpp
gko::size_type n = a.get_size();         // number of stored elements
auto           e = a.get_executor();     // shared_ptr to the executor; nullptr if default-constructed
```

## Copy and move semantics

A key rule: **copy and move assignment never change the executor of the destination array.** This makes cross-device memory movement simple and explicit.

| Operation                                   | Behaviour                                                        |
| ------------------------------------------- | ---------------------------------------------------------------- |
| `array<T> b = a;`                           | Copy. `b` shares `a`'s executor; data is copied.                 |
| `array<T> b{other_exec, a};`                | Cross-executor copy: copies `a`'s data into `other_exec`.       |
| `array<T> b = std::move(a);`                | Move construction. `a` becomes empty; no data is copied.         |

The executor-conservation rule means that assigning across executors performs a cross-device copy automatically:

```cpp
gko::array<double> a_host(host_exec, {1.0, 2.0, 3.0});
gko::array<double> a_gpu(gpu_exec, 3);   // gpu array, uninitialised
a_gpu = a_host;   // copies host → device; a_gpu remains on gpu_exec
```

## Cross-executor transfers

The canonical way to move data from host to device (or back) is via the cross-executor constructor:

```cpp
auto a_host = gko::array<double>(host_exec, {1.0, 2.0, 3.0});
auto a_gpu  = gko::array<double>(gpu_exec, a_host);   // copy host → device
```

A move assignment keeps the data only when both arrays use the same executor; moving into an array on a different executor copies the data there.

For matrices and other cloneable objects held by pointer, `gko::clone(gpu_exec, mtx)` performs the same kind of deep copy (see [Cloning](linop-and-composition.md#cloning)).

To change the executor associated with an existing array, use `set_executor`. If the new executor differs from the current one, the data is automatically migrated:

```cpp
gko::array<double> a(omp_exec, {1.0, 2.0, 3.0});
a.set_executor(hip_exec);   // values [1.0, 2.0, 3.0] are now in GPU memory
```

## Const-correctness

Prefer `const array<T>` over `array<const T>`. A `const array<T>` disables `get_data()`, `fill()`, `resize_and_reset()`, and `set_executor()`, enforcing the read-only contract at compile time. `array<const T>` is not supported.

## Resizing and filling

Resize an array and discard its contents with `resize_and_reset`:

```cpp
a.resize_and_reset(new_size);   // content is undefined after this call
```

:::{attention}
`resize_and_reset` throws if the array has no executor (default-constructed without one) or if the array is a non-owning view. See [Views and zero-copy wrapping](views-zero-copy.md) for the view restriction.
:::

Reset the size to zero:

```cpp
a.clear();   // get_data() returns nullptr afterward
```

Set every element to the same value with `fill`:

```cpp
gko::array<double> a(exec, 10);
a.fill(1.5);   // all 10 elements are now 1.5
```

`fill` is only available for the following storage types:

- Real and complex floating-point: `float`, `double`, `std::complex<float>`, `std::complex<double>`
- Index and size types: `int32`, `int64`, `size_type`

Using `fill` with any other type results in a link-time error. This restriction exists because `fill` dispatches to a GPU kernel that must be instantiated at compile time for each supported type.

## What array does not do

`array` provides raw contiguous storage, not arithmetic. It does not support element-wise addition, scaling, or dot products. For arithmetic on numeric vectors, use `gko::matrix::Dense<T>`, a `LinOp` built on top of `array` that exposes full linear algebra operations.

`array` does not support copy-on-write semantics, reference counting of the stored data, or RAII for non-POD types. For owning `LinOp` objects, use `std::shared_ptr<gko::LinOp>`.

## Wrapping memory you do not own

Owning memory is the common case, but `array` also supports a non-owning *view*
mode that wraps memory you already control (a buffer from deal.II, MFEM, Kokkos,
or a raw user pointer) without copying it. The lifetime contract, integration
patterns (deal.II distributed vector, raw user buffers, Kokkos `View`,
zero-copy CSR construction), and "when to copy instead of view" guidance live on
their own page:

```{toctree}
:maxdepth: 1
views-zero-copy
```

:::{seealso}
- [The Executor model](executor-model.md): what an `array` is bound to.
- API reference: [`gko::array`](../../api-reference/base/array.md)
:::
