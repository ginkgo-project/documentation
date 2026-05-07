# IDR(s) — Induced Dimension Reduction

*This page is a placeholder. Until it is written, the API reference covers the type:*
[`gko::solver::Idr`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1solver_1_1Idr.html)*.*

IDR(s) is a short-recurrence Krylov method for general non-symmetric systems. The
implementation follows Van Gijzen and Sonneveld, *Algorithm 913: An elegant IDR(s)
variant that efficiently exploits biorthogonality properties*. The shadow-space
dimension `s` (factory parameter `subspace_dim`, default 2) controls the trade-off
between work per iteration and convergence speed; common choices are `s = 2`, `4`,
or `8`. Unlike restarted GMRES, IDR(s)'s memory cost stays bounded as the
iteration progresses.
