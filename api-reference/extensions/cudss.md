# `gko::ext::cuda::solver::Cudss`

Wraps NVIDIA's cuDSS sparse direct solver as a Ginkgo `LinOp`. The
factorisation is computed during `generate()` and reused across
subsequent `apply()` calls; the factorisation data is opaque and held
inside the cuDSS handle.

Only runs on `CudaExecutor`, and only supports `float`, `double`,
`std::complex<float>`, and `std::complex<double>` value types. The
`matrix_type` and `matrix_view` factory parameters must be set
consistently with how the input CSR is stored; see the parameter
docs below for the details and the `matrix_type` warning.

For an introduction to the extension, see the
[cuDSS extension page](../../user-guide/concepts/extensions/cudss.md).

```{doxygenclass} gko::ext::cuda::solver::Cudss
:members:
```
