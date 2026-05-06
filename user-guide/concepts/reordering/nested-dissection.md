# Nested dissection

*This page is a placeholder. Until it is written, the API reference covers the type:*
[`gko::experimental::reorder::NestedDissection`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1experimental_1_1reorder_1_1NestedDissection.html)*.*

Nested dissection recursively partitions the matrix's adjacency graph and orders the
separators last. It tends to outperform AMD on large problems with strong locality
(finite-element, finite-difference, mesh-based applications) and is the preferred
reordering for parallel sparse direct factorisation. Requires a METIS installation
(`-DGINKGO_BUILD_METIS=ON`).
