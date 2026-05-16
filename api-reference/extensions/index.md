# `gko::ext`

Optional integrations with external libraries. Code in `gko::ext` is
opt-in — every entry here requires the corresponding third-party
dependency at build time, and falls out of the build entirely when
that dependency is not found.

For the user-facing walk-through of each extension, see
[Extensions](../../user-guide/concepts/extensions.md).

## Available extensions

- [`Cudss`](cudss.md) — wraps NVIDIA's cuDSS sparse direct solver as
  a Ginkgo `LinOp`. Lives in `gko::ext::cuda::solver` and requires
  a CUDA build with cuDSS available.
- [`Kokkos interop`](kokkos.md) — helpers for round-tripping
  Ginkgo's `array`, `Dense`, and executor types through Kokkos views
  and execution spaces. Lives in `gko::ext::kokkos`; requires Kokkos
  at build time.

```{toctree}
:hidden:
:maxdepth: 1

cudss
kokkos
```
