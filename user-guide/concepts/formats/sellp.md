# Sellp (Slice-ELLPACK)

Sliced ELLPACK is ELL's answer to non-uniform row lengths. Instead of padding
every row to the **globally** longest one, Sellp partitions the matrix into
fixed-size **slices** of contiguous rows and pads each slice to its own longest
row. A single outlier no longer inflates the storage of every other row in the
matrix — only the slice it lives in. The default slice size is 64, which
keeps memory access coalesced within a slice while shrinking padding
overall.

## Storage layout

The rows are grouped into slices of size $S$ (default 64). For slice $s$
spanning rows $[sS,\, (s+1)S)$, let $k_s$ be the longest non-zero count in
that slice. Sellp stores, per slice, two column-major arrays of size
$S \times k_s$ — the values and column indices of every entry, padded so all
rows in the slice present $k_s$ slots. A `slice_sets` prefix-sum array
records each slice's offset in **slot-column** units — slice $s$ occupies the
flat range $[\mathit{slice\_sets}[s] \cdot S,\, \mathit{slice\_sets}[s+1] \cdot S)$
of the values / col-indices buffers — and a `slice_lengths` array records each
$k_s$.

```text
Matrix (n_rows = 4):       Two slices of S = 2:
[ 1  2  0 ]                Slice 0 covers rows [0, 1]:  k_0 = 2
[ 0  3  0 ]                  values    = [1, 3,  2, 0]
[ 4  0  5 ]                  col_idxs  = [0, 1,  1, *]
[ 0  6  0 ]                Slice 1 covers rows [2, 3]:  k_1 = 2
                             values    = [4, 6,  5, 0]
                             col_idxs  = [0, 1,  2, *]
```

The `slice_sets` array would be `[0, 2, 4]` (slice 0 spans slot-columns 0..2,
slice 1 spans 2..4 — multiplied by $S=2$ this gives flat extents 0..4 and
4..8 in the values / col-indices buffers), and `slice_lengths` would be
`[2, 2]`. (* = padding: value = 0, col_idx = invalid_index<IndexType>().)

## When to use Sellp

- **Non-uniform but locally similar row lengths** — typical of unstructured
  finite-element matrices where row degree varies across the mesh but is
  similar between adjacent rows after a band-reducing reordering (see
  [RCM](../reordering/rcm.md)).
- **GPU SpMV where ELL would waste too much space.**  Each slice still gives
  coalesced access within itself; only the slices that need it grow wide.
- **As a fall-back from ELL** when row-length variance becomes too costly,
  before reaching for the heavier [Hybrid](hybrid.md) split.

The slice size $S$ is configurable; the default of 64 is a multiple of the GPU warp
width (32 on NVIDIA, 64 on AMD), and smaller slices reduce padding at the
cost of slightly worse coalescing.

## Construction

```cpp
// From matrix_data — slice_size defaults to 64
auto sellp = gko::matrix::Sellp<double, int>::create(exec);
sellp->read(data);

// With an explicit slice size (and slice-stride factor for alignment)
auto sellp_sized = gko::matrix::Sellp<double, int>::create(
                 exec,
                 gko::dim<2>{n_rows, n_cols},
                 /*slice_size=*/ 32,
                 /*stride_factor=*/ 1,
                 /*total_cols=*/ 0);   // 0 = auto

// Converted from CSR
auto sellp_csr = gko::matrix::Sellp<double, int>::create(exec);
sellp_csr->copy_from(csr);
```

A re-ordering step (e.g. RCM) before conversion can dramatically reduce the
per-slice maxima and thus the storage overhead.

## Performance characteristics

Sellp's per-output-row SpMV cost is $O(k_s)$ memory traffic for the slice that
row belongs to.  Compared with ELL's globally-uniform $k$, the average traffic
drops to roughly $\sum_s k_s / n_\text{slices} \times S$ — proportional to the
**slice-wise** maximum rather than the global one.

The break-even versus ELL is roughly: if the slice-wise maxima are within a
small factor of the average row length, Sellp shines.  If the row-length
distribution is essentially flat, plain ELL's simpler kernel is usually a hair
faster.  If the distribution has a hard heavy tail (one or two rows with
thousands of entries while the rest have a handful), Sellp's slice padding can
still amplify; reach for [Hybrid](hybrid.md) instead.

## Conversion behaviour

Sellp converts to and from CSR and Dense; ELL and COO are reached through CSR:

```cpp
// Sellp → CSR
auto csr = gko::matrix::Csr<double, int>::create(exec);
csr->copy_from(sellp);

// CSR → Sellp
auto sellp2 = gko::matrix::Sellp<double, int>::create(exec);
sellp2->copy_from(csr);
```

`copy_from(csr)` re-slices the matrix using the current slice size and stride
factor; if those have not been set, the defaults are used.

:::{seealso}
- [Matrix formats — overview](../matrix-formats.md) — when to pick Sellp vs alternatives.
- [ELL](ell.md) — single-padding-width predecessor; faster but more wasteful when row lengths vary.
- [Hybrid](hybrid.md) — ELL main part + COO overflow for heavy-tailed row lengths.
- [RCM](../reordering/rcm.md) — band-reducing reordering that often improves Sellp's per-slice maxima.
- API reference: [`gko::matrix::Sellp`](../../../api-reference/matrix/sellp.md).
:::
