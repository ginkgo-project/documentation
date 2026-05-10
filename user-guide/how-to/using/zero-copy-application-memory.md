# Zero-copy from application memory

:::{note}
Stub — content forthcoming.
:::

Wrap an application-owned CSR / Dense buffer with a Ginkgo `LinOp` without
copying. Covers `gko::array::view`, `matrix::Csr::create_const`,
`matrix::Dense::create`, lifetime requirements, and what changes when the
buffer lives in device memory.
