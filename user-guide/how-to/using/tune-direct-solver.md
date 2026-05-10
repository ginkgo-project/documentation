# Tune a sparse direct solve

:::{note}
Stub — content forthcoming.
:::

The full preprocessing recipe — `Mc64` matching, fill-reducing reordering
(`Amd` / `Rcm` / `NestedDissection`), `Lu` or `Cholesky` factorisation, and
solving with `Direct` — plus how to reuse a precomputed
`symbolic_factorization` across solves with the same sparsity pattern. Notes
on when to switch to the [cuDSS extension](../../concepts/extensions/cudss.md).
