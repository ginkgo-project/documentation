# Add a new kernel

:::{note}
Stub — content forthcoming.
:::

Walk through the full kernel-dispatch chain: declare in
`core/<area>/<type>_kernels.hpp`, register with `GKO_REGISTER_OPERATION`,
implement on `reference/`, and provide either a backend-portable
`common/unified/` implementation or backend-specific `cuda/` / `hip/` /
`omp/` / `dpcpp/` versions. Ends with the reference-parity test that
proves the new kernel matches across backends.
