# Add a new preconditioner

:::{note}
Stub — content forthcoming.
:::

The simpler preconditioner skeleton (no workspace, plain `generate()` →
`apply()`), plus the distributed variant pattern modelled on `Schwarz`:
`EnableLinOp` over a value-type-templated class, `apply_impl` calling into
local solvers via `gko::detail::get_local`, and the `Mc64`-style
configuration registration.
