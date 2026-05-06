# Reordering and permutations

A reordering computes a permutation of the rows and columns of a sparse matrix that
makes downstream operations cheaper or more numerically robust. The most common
motivation is reducing **fill-in** during sparse direct factorisation (LU,
Cholesky), but reorderings also reduce bandwidth for triangular solves, expose
diagonal dominance for preconditioning, and balance work across processes. This page
surveys Ginkgo's reordering algorithms, explains the common interface, and points to
per-algorithm pages for detail.

## What a reordering produces

Every Ginkgo reordering algorithm is a `LinOpFactory` whose `generate(matrix)` step
returns a permutation. The permutation can then be applied to the matrix
(`permuted = P * A * P^T`) and to the right-hand side / solution as needed:

```cpp
auto reorder_factory = gko::experimental::reorder::Amd<int>::build().on(exec);
auto perm = reorder_factory->generate(system_matrix);

// Apply: row+column permutation for a symmetric system
auto permuted = system_matrix->permute(perm.get());
```

For composing a reordering with a downstream solver in one shot, see `ScaledReordered`
below.

## The available algorithms

| Algorithm | Class | Goal | Typical use |
|---|---|---|---|
| RCM | `gko::experimental::reorder::Rcm` | Bandwidth / profile reduction | Preconditioning, banded triangular solves |
| AMD | `gko::experimental::reorder::Amd` | Minimise fill-in (approximate minimum degree) | Sparse direct factorisation |
| MC64 | `gko::experimental::reorder::Mc64` | Maximise diagonal product (matching) | Numerical stability for direct solvers |
| Nested dissection | `gko::experimental::reorder::NestedDissection` | Recursive bisection-based ordering | Large finite-element / FD systems on parallel hardware |
| ScaledReordered | `gko::experimental::reorder::ScaledReordered` | Combine row/column scaling and a reordering as a single LinOp wrapper around an inner solver | Apply a fill-reducing reordering transparently inside a Direct or Krylov solver |

All five live in `gko::experimental::reorder`. The non-experimental
`gko::reorder::Rcm` still exists but is **deprecated** in favour of the experimental
version, which integrates more cleanly with the rest of the reordering API.

:::{note}
`Mc64` is technically a *scaling and matching*, not just a permutation — it produces
both a permutation and a pair of diagonal scaling matrices that together push large
elements onto the diagonal. It is most useful as a preprocessing step before a direct
solver, where it improves pivoting stability.
:::

:::{note}
`NestedDissection` requires a METIS installation discoverable at build time
(`-DGINKGO_BUILD_METIS=ON`).
:::

## Why reorder before a direct solver

Sparse LU and Cholesky factorisations introduce **fill-in**: zeros in the original
matrix that become non-zero in `L` or `U`. The amount of fill-in depends strongly on
the row/column ordering. A naïve ordering can produce a factorisation many orders of
magnitude larger than the original matrix; a fill-reducing reordering (AMD, nested
dissection) typically keeps the factorisation tractable.

The recommended pre-processing recipe for a sparse direct solver is:

1. **Match (optional)** — apply MC64 to push large entries onto the diagonal.
2. **Reorder** — apply AMD or nested dissection to reduce fill-in.
3. **Factorise** — pass the permuted matrix to the LU/Cholesky factorisation.
4. **Solve** — `apply(b, x)` runs the factorised triangular solves.

The `ScaledReordered` linear operator wraps steps 1–3 around any inner solver so the
caller does not have to manage the permutations explicitly.

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
- API reference: [`gko::experimental::reorder`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/namespacegko_1_1experimental_1_1reorder.html)
:::
