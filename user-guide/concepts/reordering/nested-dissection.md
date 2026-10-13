# Nested dissection

Nested dissection (ND) is a recursive, divide-and-conquer fill-reducing reordering.
A balanced **vertex separator** of the matrix's adjacency graph is chosen, the two
disconnected sub-graphs are ordered (recursively, by the same procedure), and the
separator vertices are placed **last**. The recursion bottoms out on small
sub-graphs, which METIS orders itself.

The separator-last property bounds the fill-in to the union of fills inside each
sub-graph plus a contribution from the separator. For meshes with a regular
structure this gives asymptotically optimal complexity: for the 2-D 5-point stencil
on an $n$-row grid the factorisation has $O(n^{3/2})$ nonzeros and costs $O(n^2)$
work to compute — beating AMD's worst case by a polynomial factor.

## When to use nested dissection

- **Large mesh-based matrices** from finite-element, finite-difference, or
  finite-volume discretisations, where the underlying geometry produces good
  separators.
- **Parallel sparse direct factorisation** — the dissection tree exposes
  independent sub-problems that can be factorised concurrently, which AMD's flat
  ordering cannot.
- For **moderate-sized matrices** without obvious geometric structure, AMD often
  wins on both runtime and final fill — ND's setup cost is dominated by the
  partitioner.

## Using nested dissection

```cpp
auto reorder_factory =
    gko::experimental::reorder::NestedDissection<double, int>::build().on(exec);
auto perm = reorder_factory->generate(system_matrix);
auto permuted = system_matrix->permute(perm.get());
```

METIS options are passed through the `options` parameter: `with_options` takes a
`std::unordered_map<int, int>` keyed by the METIS option index (for example
`METIS_OPTION_SEED`), and options that are not set keep METIS's defaults. Ginkgo
supports 0-based numbering only.

## Implementation notes

The partitioning is delegated to [METIS](https://github.com/KarypisLab/METIS)
(`METIS_NodeND`); the quality of the resulting ordering is dominated by METIS's
separator quality. `NestedDissection` is declared when CMake finds METIS at
configure time, so METIS needs to be discoverable when Ginkgo is built: pass
`-DMETIS_DIR=<prefix>` or add the prefix to `CMAKE_PREFIX_PATH`. A found METIS is
enabled automatically.

:::{seealso}
- API reference: [`gko::experimental::reorder::NestedDissection`](../../../api-reference/reorder/nested-dissection.md).
- [Direct (LU / Cholesky)](../solvers/direct.md) — the primary consumer of an ND ordering on large problems.
- [Reordering and permutations](../reordering.md) — overview and comparison table.

Reference:
Karypis, G., Kumar, V. *A Fast and Highly Quality Multilevel Scheme for Partitioning
Irregular Graphs.* SIAM Journal on Scientific Computing, 20 (1), 359–392, 1998.
<https://doi.org/10.1137/S1064827595287997>
:::
