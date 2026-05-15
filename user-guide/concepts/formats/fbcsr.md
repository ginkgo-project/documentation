# Fbcsr (Fixed-Block CSR)

Fbcsr is CSR over **dense sub-blocks** instead of scalars. The matrix is viewed
as an $r \times r$ grid of $\mathrm{bs} \times \mathrm{bs}$ tiles, and the
block-level sparsity pattern is stored exactly like CSR — except that each
"non-zero" is a whole dense block. The format pays off whenever the matrix has
a natural block structure where a few dense sub-blocks repeat across the
sparsity pattern: finite-element matrices with multiple unknowns per node,
vector-PDE discretisations, and any matrix produced by a tensor-product
discretisation on a small per-DOF state space.

## Storage layout

Three arrays, parameterised by the fixed block size $\mathrm{bs}$ (which must
divide both `n_rows` and `n_cols`):

- **`row_ptrs`** — length $r + 1$, where $r = n_\text{rows} / \mathrm{bs}$ is
  the number of block-rows. `row_ptrs[i]` is the index of the first stored
  block in block-row $i$; `row_ptrs[r]` equals the total stored-block count.
- **`col_idxs`** — length equal to the number of stored blocks. The
  block-column index of each block, ordered block-row by block-row.
- **`values`** — length `num_blocks * bs * bs`. The dense data of each
  stored block, **column-major** within each block (`block_col_major`
  accessor — `fbcsr_kernels.cpp:53`), with blocks concatenated in the same
  order as `col_idxs`.

```text
Matrix (4 × 4, bs = 2):       Block layout (2 × 2 block grid):
[ 1  2  0  0 ]                Block (0,0) = [[1,2],[3,4]]
[ 3  4  0  0 ]                Block (1,0) = [[0,0],[7,8]]
[ 0  0  5  6 ]                Block (1,1) = [[5,6],[9,10]]
[ 7  8  9 10 ]                Block (0,1) is all-zero — not stored.

                            Fbcsr storage:
                              row_ptrs = [0, 1, 3]
                              col_idxs = [0, 0, 1]
                              values   = [ 1, 3, 2, 4,    (block (0,0) col-major)
                                           0, 7, 0, 8,    (block (1,0) col-major)
                                           5, 9, 6, 10 ]  (block (1,1) col-major)
```

The blocks themselves are dense — every entry of a stored block is held in
`values`, including any zeros inside it. That redundancy is the price for
fast block-SpMV: each block becomes a tight $\mathrm{bs} \times \mathrm{bs}$
dense matrix-vector product that vector hardware can dispatch as a single
unrolled kernel.

## When to use Fbcsr

- **Vector-valued PDEs** — solid mechanics with $\mathrm{bs} = 2$ or $3$ DOFs
  per node, fluid dynamics with $4$ or $5$ unknowns per cell, electromagnetic
  problems with $6$ field components.
- **Finite-element matrices** where the local element matrix has a fixed,
  small size and assembly is naturally block-structured.
- **Batched / strided BLAS-style computations** where the sub-blocks would
  otherwise be processed by hand-rolled unrolled kernels.

Fbcsr is **not** a good fit for matrices whose sparsity is genuinely scalar
(no block structure). Storing dense $\mathrm{bs} \times \mathrm{bs}$ blocks
when most entries are zero wastes both memory and FLOPs in proportion to the
fill ratio. Use plain [CSR](csr.md) instead.

## Construction

```cpp
// From matrix_data — block size must be specified
auto fbcsr = gko::matrix::Fbcsr<double, int>::create(
                 exec,
                 gko::dim<2>{n_rows, n_cols},
                 /*num_stored_blocks=*/ 0,
                 /*block_size=*/ 2);
fbcsr->read(matrix_data);

// From CSR — converts when the block size divides both dimensions
auto fbcsr = gko::matrix::Fbcsr<double, int>::create(
                 exec,
                 gko::dim<2>{n_rows, n_cols},
                 /*num_stored_blocks=*/ 0,
                 /*block_size=*/ 2);
fbcsr->copy_from(csr);
```

`read` and conversion both **require** the block size to be set on the target
beforehand; both `n_rows` and `n_cols` must be divisible by it
(`get_num_blocks` throws `BlockSizeError` otherwise).

## Performance characteristics

Per stored block, SpMV performs a $\mathrm{bs} \times \mathrm{bs}$ dense
matrix-vector product. The vector accesses are reused $\mathrm{bs}$ times
per block (once per output row), and the column-major block layout means a
warp can stream the block's columns sequentially — both effects increase the
arithmetic intensity compared to scalar CSR. For block sizes that match a
warp / SIMD width well (typically 4, 8, or 16 in mixed-precision builds), the
saved memory traffic on the index arrays — only `num_blocks` column indices
instead of `num_blocks * bs * bs` — is often the larger gain.

## Conversion behaviour

Fbcsr converts cleanly to and from CSR, Dense, and SparsityCsr; the
conversion preserves all stored entries (including the in-block zeros that
scalar formats would not have materialised):

```cpp
// Fbcsr → CSR — every stored block entry becomes a CSR non-zero
auto csr = gko::matrix::Csr<double, int>::create(exec);
csr->copy_from(fbcsr);

// CSR → Fbcsr — requires the target's block size to be set first
auto fbcsr = gko::matrix::Fbcsr<double, int>::create(
                 exec, gko::dim<2>{n_rows, n_cols},
                 /*num_stored_blocks=*/ 0, /*block_size=*/ 2);
fbcsr->copy_from(csr);
```

`CSR → Fbcsr` walks the CSR pattern, accumulates non-zeros into the
appropriate block tile, and stores any block that contains at least one
non-zero — so a block's in-block zeros are introduced silently. If the
scalar pattern does not align cleanly with $\mathrm{bs}$-sized blocks, the
resulting Fbcsr can have many sparsely-populated blocks; profile before
relying on the block format in that case.

:::{seealso}
- [Matrix formats — overview](../matrix-formats.md) — when to pick Fbcsr vs alternatives.
- [CSR](csr.md) — the scalar workhorse; the natural fallback when block
  structure is absent.
- API reference: [`gko::matrix::Fbcsr`](../../../api-reference/matrix/fbcsr.md).
:::
