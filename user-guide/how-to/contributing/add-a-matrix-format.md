# Add a new matrix format

:::{note}
Stub — content forthcoming.
:::

The `LinOp` + `EnableLinOp` + `ConvertibleTo` pattern for a new matrix
format. Covers `read` / `write` round-trips through `matrix_data`, the
SpMV kernel chain, and how to register the format with the configuration
layer so it can be selected at runtime.
