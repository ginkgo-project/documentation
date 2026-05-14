# API Reference

The C++ API surface of Ginkgo, organised by namespace. Each entry is
embedded from the live Doxygen XML, so the signatures, member lists, and
docstrings here match the source tree exactly.

For a less-curated, fully exhaustive view (every file, every internal
helper, full inheritance graphs) the standalone Doxygen deploy at
[`ginkgo-project.github.io/ginkgo-generated-documentation`](https://ginkgo-project.github.io/ginkgo-generated-documentation/)
remains the long-form reference. The pages here are the curated
entry points.

## Browse by namespace

- [`gko::matrix`](matrix/index.md) — sparse and dense matrix formats:
  Csr, Coo, Dense, Ell, Sellp, Hybrid, Fbcsr, SparsityCsr, and utility
  types (Identity, Diagonal, Permutation, RowGatherer).
- [`gko::solver`](solver/index.md) — Krylov solvers (CG, BiCGSTAB,
  GMRES, IDR(s), …), iterative refinement, direct solvers, multigrid,
  and triangular substitution.

More namespace pages (`gko::preconditioner`, `gko::log`,
`gko::experimental::distributed`, `gko::batch`, `gko::stop`,
`gko::reorder`, `gko::factorization`) will land in subsequent passes.

```{toctree}
:hidden:
:maxdepth: 1

matrix/index
solver/index
```
