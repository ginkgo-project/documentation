# Read and write matrices

Ginkgo has built-in I/O for the **Matrix Market** format (and a binary
variant). The same helpers work for every matrix type because they
round-trip through `matrix_data`.

## Read a Matrix Market file

```cpp
#include <ginkgo/ginkgo.hpp>
#include <fstream>

auto in = std::ifstream{"matrix.mtx"};
auto A  = gko::read<gko::matrix::Csr<double, int>>(in, exec);
```

`gko::read<T>(stream, exec)` constructs an instance of the requested
type and populates it from the stream. Use `Csr`, `Coo`, `Dense`, etc.
as the template argument.

For the **binary** Matrix Market variant (smaller, faster):

```cpp
auto in = std::ifstream{"matrix.bin", std::ios::binary};
auto A  = gko::read_binary<gko::matrix::Csr<double, int>>(in, exec);
```

`gko::read_generic<T>` detects the format from a magic prefix and picks
the matching reader.

## Write a matrix

```cpp
auto out = std::ofstream{"result.mtx"};
gko::write(out, A);
```

`write` takes any `LinOp` and serialises in textual Matrix Market
format. There is also `gko::write_binary(out, A)` for the binary
variant.

## Convert between formats on disk

Read into one type, then create the destination format and `copy_from`:

```cpp
auto coo = gko::read<gko::matrix::Coo<double, int>>(in, exec);
auto csr = gko::matrix::Csr<double, int>::create(exec);
csr->copy_from(coo);                                // COO → CSR
gko::write(out_stream, csr);
```

CSR and Dense are conversion hubs — every other format has a direct
edge to/from both. From any other pair you may need two hops (e.g.
`Ell → CSR → Hybrid`). See [Matrix formats](../../concepts/matrix-formats.md)
for the full conversion graph.

## Assemble from raw triplets

When the data is already in (row, col, value) triplet form, build a
`matrix_data` directly and read into any target type:

```cpp
gko::matrix_data<double, int> data{gko::dim<2>{n_rows, n_cols}};
for (auto [i, j, v] : your_triplets) {
    data.nonzeros.emplace_back(i, j, v);
}
data.sort_row_major();          // optional but recommended
data.remove_zeros();            // optional

auto A = gko::matrix::Csr<double, int>::create(exec);
A->read(data);
```

`matrix_data` is host-side; the `read` call moves the data onto the
matrix's executor.

## Common pitfalls

- **Coordinate base.** Matrix Market files are 1-indexed; Ginkgo's
  internal representation is 0-indexed. The `gko::read` helpers handle
  the offset for you, so don't subtract 1 manually.
- **Symmetric headers.** Matrix Market files marked `symmetric` store
  only one triangle. Ginkgo's readers expand them to the full matrix
  automatically.
- **`matrix_data` is host-only.** Build it on the host, then `read`
  into a matrix whose executor is wherever you want the data to end up.

:::{seealso}
- [Matrix data](../../concepts/matrix-data.md) — what `matrix_data` is and when to use it.
- [Matrix formats](../../concepts/matrix-formats.md) — the conversion graph.
- [Matrix Market format spec](https://math.nist.gov/MatrixMarket/formats.html).
:::
