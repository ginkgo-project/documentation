# Hybrid (ELL + COO overflow)

Hybrid splits each row's non-zeros between a fixed-width [ELL](ell.md) section
and an overflow [COO](coo.md) tail. The ELL part keeps the regular,
coalesced-friendly layout for the *common* portion of every row, while the
COO part absorbs any non-zeros that exceed the ELL width, typically a few
long rows in an otherwise short-row matrix. SpMV runs as the sum of an ELL
SpMV and a COO SpMV; both kernels still touch the same input and output
vectors, but each is well-tuned for the layout it operates on.

## Storage layout

A Hybrid matrix owns two child matrices:

- An internal **`gko::matrix::Ell`** of width $k_\text{ell}$ holding the
  first $k_\text{ell}$ non-zeros of every row (padded as usual when a row has
  fewer).
- An internal **`gko::matrix::Coo`** holding the remaining non-zeros, only
  populated for rows whose original non-zero count exceeded $k_\text{ell}$.

```text
Matrix (4 × 4, one heavy row at the bottom):
[ 1  0  2  0 ]             ELL part (k_ell = 2, column-major):
[ 0  3  0  0 ]               values   = [1, 3, 4, 6,  2, 0, 5, 7]
[ 4  0  5  0 ]               col_idxs = [0, 1, 0, 0,  2, *, 2, 1]
[ 6  7  8  9 ]
                           COO part (overflow from row 3):
                             row_idxs = [3, 3]
                             col_idxs = [2, 3]
                             values   = [8, 9]
```

The ELL part dominates storage on a typical matrix: the COO overflow holds
only the entries the ELL split could not accommodate.

## Choosing the split: strategies

The width $k_\text{ell}$ is the central knob. Pick it too small and the COO
tail grows; pick it too large and the ELL padding wastes memory. Hybrid
ships five strategies that pick $k_\text{ell}$ for you at construction time:

| Strategy | What it picks |
|---|---|
| `automatic` *(default)* | An `imbalance_bounded_limit` with fraction 1/3 and ratio 0.001. |
| `column_limit` | A fixed user-supplied column count. |
| `imbalance_limit(percent)` | A $k$ that keeps at least `percent` of the rows entirely in the ELL part (default 0.8). |
| `imbalance_bounded_limit(percent, ratio)` | Like `imbalance_limit`, but $k$ never exceeds `ratio * num_rows` (defaults 0.8 and 0.0001). |
| `minimal_storage_limit` | An `imbalance_limit` whose fraction is derived from the value and index sizes to keep the total bytes (ELL + COO) small. |

The chosen strategy is set on the Hybrid object before reading the data:

```cpp
auto hybrid = gko::matrix::Hybrid<double, int>::create(
    exec,
    std::make_shared<
        gko::matrix::Hybrid<double, int>::imbalance_limit>(0.8));
hybrid->read(data);
```

## When to use Hybrid

- **Heavy-tailed row-length distributions.**  A handful of long rows would
  inflate ELL's $k_\text{max}$ by orders of magnitude; the COO overflow
  absorbs them.
- **GPU SpMV on irregular matrices** where neither plain ELL nor
  [Sellp](sellp.md) recovers enough of ELL's coalesced access.
- **Sparse matrices arising from agglomerated meshes** with a few high-degree
  super-nodes alongside many low-degree regular nodes.

For matrices with **uniform** row lengths, plain ELL or CSR is simpler and
faster; Hybrid's two-kernel SpMV adds overhead that only pays off when the
ELL alone would waste significant memory.

## Construction

```cpp
// From matrix_data with the default automatic strategy
auto hybrid = gko::matrix::Hybrid<double, int>::create(exec);
hybrid->read(data);

// With an explicit strategy
auto hybrid_limit = gko::matrix::Hybrid<double, int>::create(
    exec,
    std::make_shared<
        gko::matrix::Hybrid<double, int>::column_limit>(/*ell_width=*/ 8));
hybrid_limit->read(data);

// Converted from CSR
auto hybrid_csr = gko::matrix::Hybrid<double, int>::create(exec);
hybrid_csr->copy_from(csr);
```

The chosen strategy is sticky on the Hybrid instance: any `copy_from` or
`read` after construction reuses the same split policy.

## Performance characteristics

The total SpMV cost is the sum of:

- An ELL SpMV over a fixed-width $k_\text{ell}$ block, typically the bulk
  of the work, executing with full coalescing.
- A COO SpMV over the overflow entries: partial sums of the same row are
  combined with a segmented scan and added to the output vector with atomics.

For matrices well-suited to Hybrid the COO part is a small fraction of the
non-zero count (often < 10 %), so its cost is dominated by the ELL kernel.
For matrices where the COO overflow approaches the ELL workload, Hybrid loses
its advantage; check the split with `get_ell_num_stored_elements()` and
`get_coo_num_stored_elements()` after construction.

## Conversion behaviour

Hybrid converts to and from CSR and Dense; COO and ELL are reached through CSR:

```cpp
// Hybrid → CSR
auto csr = gko::matrix::Csr<double, int>::create(exec);
csr->copy_from(hybrid);

// CSR → Hybrid
auto hybrid2 = gko::matrix::Hybrid<double, int>::create(exec);
hybrid2->copy_from(csr);   // uses the default automatic strategy
```

Conversion preserves all numerical entries: the COO overflow re-merges with
the ELL part to produce a flat CSR matrix.

:::{seealso}
- [Matrix formats](../matrix-formats.md): when to pick Hybrid vs alternatives.
- [ELL](ell.md): the dense underlying format when row lengths are uniform.
- [Sellp](sellp.md): per-slice padding; the natural in-between choice when row lengths vary smoothly rather than in a heavy tail.
- [COO](coo.md): the overflow representation used internally.
- API reference: [`gko::matrix::Hybrid`](../../../api-reference/matrix/hybrid.md).
:::
