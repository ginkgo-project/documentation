# COO (Coordinate)

COO stores a sparse matrix as a flat list of `(row, column, value)` triples. It is
Ginkgo's assembly-friendly format — straightforward to build incrementally, easy to
extend with new entries, and directly compatible with the Matrix Market file format.
Its SpMV is slower than CSR for repeated use, so the typical workflow is to assemble
in COO and then convert to CSR. This page covers when to use COO, its storage layout,
construction patterns, SpMV behaviour, and the conversion path.

## Storage layout

COO uses three arrays, each of length `nnz`:

- **`row_idxs[k]`** — row index of non-zero `k`.
- **`col_idxs[k]`** — column index of non-zero `k`.
- **`values[k]`** — numerical value of non-zero `k`.

```text
Matrix:                COO storage:
[ 1  0  2 ]            row_idxs = [0, 0, 1, 2, 2]
[ 0  3  0 ]            col_idxs = [0, 2, 1, 0, 2]
[ 4  0  5 ]            values   = [1, 2, 3, 4, 5]
```

Unlike CSR, COO does not compress row information into a prefix-sum array. Each non-zero
explicitly stores its row index. This makes random insertion cheap (append a triple) but
SpMV more expensive (the hardware must handle arbitrary row scatter via atomic
operations).

COO does not require entries to be sorted. `read(matrix_data)` produces row-major order,
but explicit-array construction does not enforce any ordering.

## When to use COO

**Assembly.** Building a matrix from many small contributions — finite-element element
matrices, stencil contributions, entries accumulated from distributed subdomains — is
natural in COO. Appending triples is `O(1)` per entry. No reordering or compaction is
needed during assembly.

**File IO.** Matrix Market (`.mtx`) is structurally a COO format. `gko::read_raw`
returns a `matrix_data` whose entries correspond one-to-one to COO triples. If you read
a matrix from disk and immediately convert it to CSR, COO is never explicitly
materialised — but it is the conceptual intermediate.

**Exploratory or one-shot use.** If you construct a matrix once and use it a small number
of times, the slower COO SpMV may be acceptable and you avoid the conversion cost.

## Construction

```cpp
// From matrix_data (the typical assembly path)
auto coo = gko::matrix::Coo<double, int>::create(exec);
coo->read(data);

// Direct construction from three explicit arrays
auto coo = gko::matrix::Coo<double, int>::create(
               exec,
               gko::dim<2>{n_rows, n_cols},
               std::move(values_array),
               std::move(col_idxs_array),
               std::move(row_idxs_array));
```

Note the argument order in the explicit-array constructor: `values`, then `col_idxs`,
then `row_idxs` — this matches the Ginkgo convention for sparse constructors (value
arrays before index arrays, column before row).

:::{tip}
If you are loading from a Matrix Market file and want COO, use
`gko::read<gko::matrix::Coo<double, int>>(stream, exec)` for a compact one-liner.
:::

## SpMV in COO

COO's SpMV implementation assigns one thread per non-zero and uses atomic accumulation
into the output vector:

```text
for each non-zero k (in parallel):
    atomic_add(y[row_idxs[k]], values[k] * x[col_idxs[k]])
```

This is correct on every backend but carries two performance costs:

1. **Atomic overhead.** Multiple threads writing to the same `y[i]` (rows with many
   non-zeros) serialise through the atomic, reducing parallelism.
2. **Irregular memory access.** `row_idxs[k]` addresses `y` non-sequentially, reducing
   cache effectiveness compared to CSR's row-sequential access.

On modern GPUs, atomics to distinct addresses are fast, so COO SpMV is not catastrophically
slow — but it is consistently slower than CSR on the same matrix.

## `apply2` — accumulating SpMV

Because COO's SpMV kernel already uses atomic accumulation into the output, COO offers
an extra entry point that the other matrix formats do not: `apply2`. Where the standard
`apply(b, x)` *overwrites* `x` with `A * b`, `apply2(b, x)` *adds* `A * b` to whatever
`x` already holds:

```cpp
// Standard apply (inherited LinOp interface): x = A * b
coo->apply(b, x);

// COO-specific apply2: x = A * b + x
coo->apply2(b, x);

// Generalised apply2: x = alpha * A * b + x
coo->apply2(alpha, b, x);
```

This is essentially a fused SpMV + AXPY at no extra cost — the existing atomic update
on `x` simply skips the zero-initialisation step. It is useful in algorithms that need
to accumulate contributions from several operators into the same output (for example,
the off-diagonal block in additive operator splittings).

`apply2` is declared on `gko::matrix::Coo` directly, not on the `LinOp` base — calling
it through a `LinOp` pointer requires an explicit cast (`as<gko::matrix::Coo<...>>`).

## Convert to CSR for repeated SpMV

If you perform SpMV more than a few times, the conversion cost pays off quickly:

```cpp
auto csr = gko::matrix::Csr<double, int>::create(exec);
csr->copy_from(coo);
// coo can now be released if no longer needed
```

The conversion is `O(nnz)` and runs on the executor (GPU or CPU). It performs a parallel
prefix-sum over row counts to build `row_ptrs`, then copies `col_idxs` and `values`.

## Limitations

- **No row-access semantics.** CSR's `row_ptrs` array lets algorithms skip directly to
  row `i` in `O(1)`. COO has no such shortcut — scanning for row `i` is `O(nnz)` unless
  the data is pre-sorted.
- **Larger index storage.** COO stores `nnz` row indices; CSR stores only `n_rows + 1`
  row pointers. For matrices where `n_rows << nnz`, CSR uses noticeably less memory.
- **Preconditioners and factorisation require CSR.** Passing a COO matrix to a
  preconditioner factory will trigger an implicit conversion. Pre-converting avoids the
  repeated overhead.

## Publications

- [Balanced and Compressed Coordinate Layout for the Sparse Matrix-Vector Product on GPUs](https://doi.org/10.1007/978-3-030-71593-9_7)

:::{seealso}
- [Matrix data and assembly](../matrix-data.md) — `matrix_data` reads naturally into COO.
- [CSR](csr.md) — the conversion target for repeated SpMV.
- API reference: [`gko::matrix::Coo`](../../../api-reference/matrix/coo.md)
:::
