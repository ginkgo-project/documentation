# SparsityCsr (sparsity-pattern only)

`gko::matrix::SparsityCsr<ValueType, IndexType>` stores **only the sparsity pattern**
of a sparse matrix (the row pointers and column indices in the standard CSR layout)
with no per-entry value array. Every non-zero entry is treated as having the same
shared scalar value (default `1.0`, settable through the constructor). This makes it
the natural representation for graph-style algorithms, symbolic factorisation
preparation, and any analysis that depends on *where* the non-zeros live rather than
their numerical magnitudes.

It is not part of the value-bearing matrix-format taxonomy on the
[overview page](../matrix-formats.md) for that reason; it does not store enough
information to participate in a meaningful SpMV against a general right-hand side.

## Storage layout

SparsityCsr uses two index arrays plus a single shared scalar:

- **`row_ptrs`**: length `n_rows + 1`. Same semantics as in CSR: `row_ptrs[i]` is the
  starting index (into `col_idxs`) of row `i`.
- **`col_idxs`**: length `nnz`. The column indices of the non-zeros.
- **`value`**: a single scalar (one element). Every non-zero entry is treated as
  taking this value during apply. Defaults to `1.0` if not set.

The total storage footprint is `(n_rows + 1) + nnz` index slots plus one scalar,
strictly smaller than CSR for the same sparsity pattern, since the per-entry value
array is gone.

## When to use it

SparsityCsr is appropriate when one of the following is true:

- **Graph-style analysis.** You want to reason about the matrix as the adjacency
  graph of a graph algorithm (BFS, connected components, partitioning), and the
  numerical values are irrelevant.
- **Symbolic factorisation prep.** Computing the symbolic structure of a future
  factorisation depends only on the sparsity pattern. SparsityCsr is the right
  representation to feed into such an analysis.
- **Reordering input.** Partitioners and reordering algorithms operate on the graph
  structure of the matrix. SparsityCsr (especially via `to_adjacency_matrix()`,
  see below) is the canonical form to hand to them.
- **Pattern-comparison or test scaffolding.** When you want to assert that a
  computed matrix has a specific non-zero pattern but do not care about values,
  SparsityCsr makes the comparison cheap.

If you need to apply the matrix to a vector for any non-trivial value, convert to
[CSR](csr.md) (or another value-bearing format) first.

## Construction

Two factory overloads are typical:

```cpp
// Empty SparsityCsr to be filled by `read(data)` from a matrix_data
auto sp = gko::matrix::SparsityCsr<double, int>::create(exec);
sp->read(matrix_data);

// From explicit arrays (when you already hold row_ptrs and col_idxs)
auto from_arrays = gko::matrix::SparsityCsr<double, int>::create(
              exec,
              gko::dim<2>{n_rows, n_cols},
              std::move(col_idxs_array),
              std::move(row_ptrs_array),
              /*value=*/1.0);
```

Like the other CSR-family formats, SparsityCsr accepts rvalue arrays in the
explicit constructor and moves them in; passing views of user-owned buffers
gives a zero-copy wrapper. See [Views and zero-copy wrapping](../views-zero-copy.md)
for the lifetime contract.

## Conversion to and from other formats

SparsityCsr converts directly to and from CSR and Dense, and accepts `matrix_data`
through the standard `read()` interface. Fbcsr also converts directly to SparsityCsr, which is
useful for extracting the block-level graph of a block-structured matrix:

```cpp
// SparsityCsr → CSR (every non-zero gets the shared scalar)
auto csr = gko::matrix::Csr<double, int>::create(exec);
csr->copy_from(sp);

// CSR → SparsityCsr (drops the values, keeps the structure)
auto sp2 = gko::matrix::SparsityCsr<double, int>::create(exec);
sp2->copy_from(csr);
```

The conversion is `O(nnz)`. CSR → SparsityCsr is essentially a structure-only
projection; the reverse fills every non-zero with the SparsityCsr's shared scalar.

## Adjacency-matrix view

```cpp
auto adj = sp->to_adjacency_matrix();
```

`to_adjacency_matrix()` returns a square SparsityCsr representing the *graph*
underlying the matrix: the original sparsity pattern with the diagonal removed
(self-loops dropped). The input must be square. This is the form expected by graph
libraries such as METIS for partitioning and reordering: feeding the adjacency
matrix rather than the original lets the partitioner ignore the diagonal entries
that would otherwise be treated as self-loops.

## Sorted column indices

The same considerations as [CSR](csr.md#sorted-column-indices) apply: the data
structure does not enforce sorted column indices within each row, but many
downstream algorithms (especially symbolic factorisation prep) assume sortedness.
Two helpers manage this:

```cpp
bool sorted = sp->is_sorted_by_column_index();
sp->sort_by_column_index();
```

`read(matrix_data)` produces sorted output; the explicit-array constructor does
not, so callers building SparsityCsr from raw arrays should sort if the consumer
expects it.

:::{seealso}
- [Matrix formats](../matrix-formats.md): where SparsityCsr fits relative to the value-bearing formats.
- [CSR](csr.md): the value-bearing format SparsityCsr converts to and from.
- [Reordering and permutations](../reordering.md): the typical consumer of `to_adjacency_matrix()`.
- API reference: [`gko::matrix::SparsityCsr`](../../../api-reference/matrix/sparsity-csr.md)
:::
