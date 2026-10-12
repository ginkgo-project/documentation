# `gko::PolymorphicObject`

The base of every Ginkgo object. It holds the executor and provides
`validate_data()`. That function checks an object's stored data and throws
[`gko::InvalidData`](exceptions.md) on a violation. Ginkgo never calls it on
its own; call it after filling a matrix from external data.

The checks depend on the format:

| Format | What `validate_data()` checks |
|---|---|
| `matrix::Csr` | row pointers are non-descending and within `[0, nnz]`; column indices are within `[0, num_cols)` and unique within each row; all values are finite |
| `matrix::Coo` | row indices are non-descending and within `[0, num_rows)`; column indices are within `[0, num_cols)`; all values are finite |
| `matrix::Ell` | the column indices of non-padding entries are below `num_cols` and unique within each row; the stored values are finite |
| `matrix::Dense`, `matrix::Diagonal` | all values are finite |
| `matrix::Permutation` | the index array is a bijection on `[0, n)` |

Other types keep the default, which accepts the object. The check copies the
data to the host and scans it, so it costs a device-to-host transfer and
$O(\text{nnz})$ work.

```cpp
auto A = gko::matrix::Csr<double, int>::create(exec, gko::dim<2>{3}, 4);
// ... fill A's row pointers, column indices and values from external data
try {
    A->validate_data();
} catch (const gko::InvalidData& e) {
    std::cerr << "invalid matrix: " << e.what() << '\n';
}
```

```{doxygenclass} gko::PolymorphicObject
:members:
```

:::{seealso}
- [Exceptions](exceptions.md)
- [`gko::Cloneable`](cloneable.md)
:::
