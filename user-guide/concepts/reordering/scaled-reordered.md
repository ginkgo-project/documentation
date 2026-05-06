# ScaledReordered

*This page is a placeholder. Until it is written, the API reference covers the type:*
[`gko::experimental::reorder::ScaledReordered`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1experimental_1_1reorder_1_1ScaledReordered.html)*.*

`ScaledReordered` is a `LinOp` wrapper that bundles row/column scaling and a
reordering with an inner solver. The user supplies an inner solver factory (typically
`Direct` or a Krylov solver) and optionally a row scaling, a column scaling, and a
reordering factory; the wrapper handles applying the permutations and the scalings
around the inner solve so that the caller can pass in the original system matrix and
right-hand side without manual transformations.
