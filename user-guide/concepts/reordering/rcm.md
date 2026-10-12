# RCM (Reverse Cuthill–McKee)

Reverse Cuthill–McKee (RCM) is a bandwidth-reducing reordering. It walks the
matrix's adjacency graph by breadth-first search starting from a *pseudoperipheral*
vertex (one of high eccentricity, computed by an inner BFS), enumerates the
resulting level sets in order, and then **reverses** the resulting permutation. The
reversal is the key step — it consistently tightens the *profile* of the matrix even
when it does not change the worst-case bandwidth. The output is a row/column
permutation that clusters nonzeros tightly along the diagonal.

## When to use RCM

- **Triangular solves and ILU-style preconditioners on CPU.** Compacting the
  bandwidth shortens dependency chains in `LowerTrs`/`UpperTrs` and improves cache
  reuse in `Ilu` / `ParIlu`.
- **Banded direct solvers and out-of-core factorisation** where storage is dominated
  by the bandwidth.
- As a **cheap preprocessing step** before a more elaborate reordering — RCM's BFS
  is essentially linear in the number of nonzeros, so it is far cheaper to compute
  than AMD or nested dissection.

RCM is **not** a good fill-reducing reordering on its own. For sparse direct
factorisation prefer [AMD](amd.md) or [nested dissection](nested-dissection.md).

## Using RCM

```cpp
auto reorder_factory =
    gko::experimental::reorder::Rcm<int>::build().on(exec);
auto perm = reorder_factory->generate(system_matrix);

auto permuted = system_matrix->permute(perm.get());
```

RCM produces a symmetric permutation; for non-symmetric matrices it operates on the
symmetrised adjacency pattern $A + A^T$. The factory parameters are:

| Parameter | Type | Purpose |
|---|---|---|
| `skip_symmetrize` | `bool` | Use the pattern of $A$ directly instead of $A + A^T$; default `false`. |
| `strategy` | `rcm_starting_strategy` | Starting-vertex choice: `pseudo_peripheral` (default) or `minimum_degree`. |

## Implementation notes

The algorithm runs on the factory's executor (Reference, OpenMP, CUDA and HIP; SYCL
uses the host). By default the starting vertex of each connected component is a
pseudo-peripheral vertex found by repeated BFS; `with_strategy(gko::experimental::reorder::rcm_starting_strategy::minimum_degree)`
starts from a minimum-degree vertex instead.

:::{seealso}
- API reference: [`gko::experimental::reorder::Rcm`](../../../api-reference/reorder/rcm.md).
- [ILU preconditioner](../preconditioners/ilu.md) and [IC preconditioner](../preconditioners/ic.md) — typical consumers of an RCM pre-step on CPU.
- [Reordering and permutations](../reordering.md) — overview and comparison table.
:::
