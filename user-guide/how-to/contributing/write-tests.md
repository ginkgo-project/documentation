# Write tests

:::{note}
Stub — content forthcoming.
:::

The reference-parity test pattern: run on the `Reference` executor and on
the target backend, compare results within a chosen tolerance. Covers the
existing GoogleTest fixtures, matrix generators, and the conventions for
splitting unit tests across `core/test/`, `reference/test/`, and the
backend-specific `cuda/test/` / `hip/test/` / `dpcpp/test/` trees.
