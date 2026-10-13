# Reordering and permutations

A reordering computes a permutation of the rows and columns of a sparse matrix that
makes downstream operations cheaper or more numerically robust. The most common
motivation is reducing **fill-in** during sparse direct factorisation (LU,
Cholesky), but reorderings also reduce bandwidth for triangular solves, expose
diagonal dominance for preconditioning, and balance work across processes. This page
surveys Ginkgo's reordering algorithms, explains the common interface, and points to
per-algorithm pages for detail.

## What a reordering produces

`Amd`, `Rcm` and `NestedDissection` are `LinOpFactory` classes whose
`generate(matrix)` step returns a `gko::matrix::Permutation`. `Mc64` returns a
`gko::Composition` of two `gko::matrix::ScaledPermutation` objects (see
[MC64](reordering/mc64.md)). A `Permutation` can be applied to the matrix
(`permuted = P * A * P^T`) and to the right-hand side / solution as needed; `Csr`
and `Dense` provide `permute`:

```cpp
auto reorder_factory = gko::experimental::reorder::Amd<int>::build().on(exec);
auto perm = reorder_factory->generate(system_matrix);

// Apply: row+column permutation for a symmetric system
auto permuted = system_matrix->permute(perm.get());
```

To wrap diagonal scalings and an RCM reordering around an inner solver, see
`ScaledReordered` below.

## Permutation modes

Each matrix format's `permute(perm, mode)` member takes a `permute_mode` enum that
controls **how** the permutation is applied. The enum is a 3-bit mask combining
*rows*, *columns*, and *inverse*:

| Mode | Effect on `A` | Typical use |
|---|---|---|
| `permute_mode::none` | $A' = A$ | No-op (mask cleared). |
| `permute_mode::rows` | $A' = P A$ | Permute rows only. |
| `permute_mode::columns` | $A' = A P^T$ | Permute columns only. |
| `permute_mode::symmetric` *(default)* | $A' = P A P^T$ | Symmetric reordering — the standard pre-factorisation transform. |
| `permute_mode::inverse_rows` | $A' = P^{-1} A$ | Undo a previous row permutation. |
| `permute_mode::inverse_columns` | $A' = A P^{-T}$ | Undo a previous column permutation. |
| `permute_mode::inverse_symmetric` | $A' = P^{-1} A P^{-T}$ | Undo a symmetric reordering (e.g., to map a solution computed in the permuted system back to the original variable order). |

The fundamental modes can be combined with the bitwise `|`, `&`, and `^` operators —
`rows | inverse` is the same as `inverse_rows`, and so on.

`gko::matrix::Permutation<IndexType>` exposes two more helpers for working with
permutations directly:

```cpp
auto inv  = perm->compute_inverse();         // P^{-1}
auto comp = perm1->compose(perm2);           // P_2 * P_1 — apply perm1 first, then perm2
```

`compute_inverse()` returns `P^{-1}` as a new `Permutation` object. `compose(other)`
combines two permutations so that applying the result is equivalent to applying
`this` first and then `other` — useful when you want to chain a fill-reducing
reordering with a separate parallel-partitioning reordering, for instance.

For non-symmetric problems where rows and columns should be permuted by **different**
permutations (for example two separate reorderings, one supplying the row
permutation and one the column permutation), use the two-permutation overload of
`permute`:

```cpp
auto permuted = matrix->permute(row_perm.get(), col_perm.get());
```

## The available algorithms

| Algorithm | Class | Goal | Typical use |
|---|---|---|---|
| RCM | `gko::experimental::reorder::Rcm` | Bandwidth / profile reduction | Preconditioning, banded triangular solves |
| AMD | `gko::experimental::reorder::Amd` | Minimise fill-in (approximate minimum degree) | Sparse direct factorisation |
| MC64 | `gko::experimental::reorder::Mc64` | Maximise diagonal product (matching), with scaling | Numerical stability for direct solvers |
| Nested dissection | `gko::experimental::reorder::NestedDissection` | Recursive bisection-based ordering | Large finite-element / FD systems on parallel hardware |
| ScaledReordered | `gko::experimental::reorder::ScaledReordered` | Wrap fixed diagonal row/column scalings and an RCM reordering around an inner operator | Apply a bandwidth-reducing reordering transparently around a Krylov or Direct solver |

All five live in `gko::experimental::reorder`. The class template
`gko::reorder::Rcm<ValueType, IndexType>` is **deprecated** in favour of
`gko::experimental::reorder::Rcm<IndexType>`, which integrates more cleanly with the
rest of the reordering API.

:::{attention}
`ScaledReordered` takes its `reordering` factory from the older `gko::reorder`
interface, whose only implementation is the deprecated `gko::reorder::Rcm`. See [ScaledReordered](reordering/scaled-reordered.md).
:::

:::{note}
`Mc64` is a *scaling and matching*, not just a permutation. With the default
`max_diagonal_product` strategy it returns a row permutation with row and column
scaling coefficients that together push large elements onto the diagonal. It is
most useful as a preprocessing step before Ginkgo's `Lu`, which factorises without
pivoting: MC64 makes that factorisation stable.
:::

:::{note}
`NestedDissection` is available when CMake finds METIS at configure time. Point CMake
at an installation with `-DMETIS_DIR=<prefix>` or `CMAKE_PREFIX_PATH`; a found METIS
is enabled automatically.
:::

## Why reorder before a direct solver

Sparse LU and Cholesky factorisations introduce **fill-in**: zeros in the original
matrix that become non-zero in `L` or `U`. The amount of fill-in depends strongly on
the row/column ordering. A naïve ordering can produce a factorisation many orders of
magnitude larger than the original matrix; a fill-reducing reordering (AMD, nested
dissection) typically keeps the factorisation tractable.

The recommended pre-processing recipe for a sparse direct solver is:

1. **Match (optional)** — compute MC64 and apply its scaled permutations with
   `Csr::scale_permute`.
2. **Reorder** — apply the AMD or nested dissection permutation with `Csr::permute`.
3. **Factorise** — pass the permuted matrix to the LU/Cholesky factorisation.
4. **Solve** — permute `b` in the same way as the rows of the matrix, call
   `apply(b_permuted, y)` on the factorised solver, and map `y` back to the original
   variable order with the inverse permutation to obtain `x`. When step 1 uses MC64's
   `max_diagonal_product` strategy, `y` is also mapped back through MC64's column
   `ScaledPermutation`, which applies the column scaling $D_c$.

The `ScaledReordered` linear operator manages this bookkeeping for fixed diagonal
scalings and an RCM reordering around an inner solver; for MC64, AMD and nested
dissection, apply the transforms with `Csr::scale_permute` and `Csr::permute` as above.

## When to reorder for iterative solvers

Reordering helps Krylov methods less directly, but is still useful in two scenarios:

- **As a preprocessing step for an ILU/IC preconditioner** — incomplete factorisations
  also benefit from fill-reducing reorderings; an RCM or AMD pass before computing
  the preconditioner can substantially change its quality.
- **For cache locality on long triangular solves** — RCM compacts the bandwidth of the
  matrix, which can speed up the `LowerTrs` / `UpperTrs` kernels used inside ILU and
  direct solves on CPU.

## Per-algorithm pages

```{toctree}
:maxdepth: 1
:caption: Reordering algorithms

reordering/rcm
reordering/amd
reordering/mc64
reordering/nested-dissection
reordering/scaled-reordered
```

:::{seealso}
- [Direct (LU / Cholesky)](solvers/direct.md) — the primary consumer of fill-reducing reorderings.
- [Solvers — taxonomy](solvers.md) — where reordering fits relative to the solver families.
- API reference: [`gko::experimental::reorder`](../../api-reference/reorder/index.md).
:::
