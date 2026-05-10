# Move data between executors

:::{note}
Stub — content forthcoming.
:::

When to use `clone(exec)` vs `copy_from` vs zero-copy `create_const`, with
worked examples for `Dense`, `Csr`, and plain `gko::array`. Includes the
common pitfall of operating on a `LinOp` whose executor disagrees with the
operands.
