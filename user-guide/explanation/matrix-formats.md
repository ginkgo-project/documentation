# Matrix formats — an overview

Ginkgo supports several sparse-matrix storage formats, each tuned for a different memory
access pattern and workload. This page introduces the format zoo, explains the trade-off
axes that separate them, describes the conversion graph, and offers selection guidance.
Detailed coverage of individual formats lives in the `formats/` sub-pages.

## The format zoo

| Format | Class | Storage idea | Best for |
|---|---|---|---|
| CSR | `gko::matrix::Csr` | Row-pointers + col-indices + values | General-purpose default, fast SpMV |
| COO | `gko::matrix::Coo` | Triples (row, col, value) | Assembly, file IO |
| ELL | `gko::matrix::Ell` | Padded fixed-width-per-row | Matrices with near-uniform row length |
| Sellp | `gko::matrix::Sellp` | Slab-padded ELL (slice-ELL) | GPU-friendly for moderately-irregular rows |
| Hybrid | `gko::matrix::Hybrid` | ELL part + COO overflow | Mixed-row-length matrices |
| Fbcsr | `gko::matrix::Fbcsr` | Block-CSR (small dense blocks) | Blocked structures (e.g., FE matrices) |
| Dense | `gko::matrix::Dense` | Row-major dense | Multi-vectors, small dense matrices |

All seven implement the `LinOp` interface: every format supports `apply(b, x)` for SpMV,
and most support `copy_from` conversions to and from the other formats.

## Trade-off axes

The table above reduces each format to one line. Five axes give a fuller picture:

**Assembly cost.** How cheaply can you build the matrix from random `(row, col, value)`
triples? COO wins — it stores triples directly. CSR requires a prefix-sum over row
counts; ELL and Sellp require knowing the longest row up front. If your application
assembles incrementally (adding contributions from element matrices), use COO or
`matrix_data` first, then convert.

**SpMV throughput.** SpMV is memory-bound for typical sparse matrices. CSR is the
workhorse on CPU. ELL and Sellp can outperform CSR on GPU for matrices with
near-uniform row lengths because they eliminate the indirect row-pointer read.
Hybrid covers the case where some rows are long outliers.

**Memory footprint.** ELL pads every row to the same length as the longest row —
catastrophic for highly skewed row-length distributions. Sellp mitigates by grouping
rows into slabs and padding only within a slab. Hybrid offloads overflow rows to COO.

**Conversion cost.** Converting between formats is a device kernel — fast, but not free.
Most conversions route through CSR (see next section). If you convert repeatedly inside
a tight loop, profile whether the conversion dominates.

**Operation coverage.** CSR has the broadest kernel coverage in Ginkgo: SpMV,
transpose, scaling, factorisation. Other formats cover SpMV and a subset of the rest.
Preconditioners and factorisation algorithms typically require CSR input and will
convert silently if given another format.

## The conversion graph

Most format-to-format conversions route through CSR:

```
matrix_data ──► COO ──► CSR ──► ELL
                         │ ──► Sellp
                         │ ──► Hybrid
                         │ ──► Fbcsr
                         └──► Dense  (small matrices only)
```

CSR ↔ COO conversion is cheap (both store one index per non-zero; the difference is
compressed vs explicit row indices). Dense ↔ sparse is only sensible for matrices small
enough to fit in dense memory.

The conversion API is uniform across all formats:

```cpp
auto coo = gko::matrix::Coo<double, int>::create(exec);
coo->read(data);

// COO → CSR
auto csr = gko::matrix::Csr<double, int>::create(exec);
csr->copy_from(coo);

// CSR → ELL
auto ell = gko::matrix::Ell<double, int>::create(exec);
ell->copy_from(csr);

// CSR → Sellp
auto sellp = gko::matrix::Sellp<double, int>::create(exec);
sellp->copy_from(csr);
```

`copy_from` runs on the matrix's executor and returns the matrix in the new format.
The source matrix is unchanged.

:::{note}
Some conversions allocate temporary intermediate storage. For very large matrices on a
memory-constrained GPU, prefer a direct path (e.g., COO → CSR) over a multi-hop path.
:::

## Selection guidance

When choosing a format, work through these questions in order:

1. **Default to CSR.** It performs well on both CPU and GPU, has the most kernel coverage
   in Ginkgo, and is what preconditioners and factorisation algorithms expect.

2. **Use COO for assembly, then convert.** Building a matrix incrementally from element
   contributions is easiest in COO (or `matrix_data`). Construct COO, call `csr->copy_from(coo)`,
   and release the COO when you no longer need it.

3. **Consider ELL or Sellp for GPU with regular structure.** If your matrix comes from a
   structured mesh or a banded problem where every row has roughly the same number of
   non-zeros, ELL or Sellp can outperform CSR in repeated SpMV on GPU. Sellp is more
   flexible (it handles moderate variation with less padding waste than ELL).

4. **Use Hybrid for mixed-row-length matrices.** Typical example: coarse-grid levels in
   algebraic multigrid, where most rows are short but a few are long. Hybrid stores the
   short rows in ELL and overflows long rows into COO.

5. **Use Fbcsr for block-structured matrices.** If your matrix has a natural block
   structure — for example, `k` degrees of freedom per finite-element node, producing
   `k×k` dense blocks — Fbcsr stores those blocks explicitly and exploits their dense
   structure in SpMV.

6. **Dense is not a sparse format.** Use `gko::matrix::Dense` for multi-vectors (the
   right-hand side `b` and solution `x` in SpMV) and for small dense matrices. Do not
   use it to store a large sparse matrix.

:::{seealso}
- [Matrix data and assembly](matrix-data.md) — getting data into a format.
- [CSR](formats/csr.md), [COO](formats/coo.md) — the two filled format pages.
- API reference: [`gko::matrix`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/namespacegko_1_1matrix.html)
:::
