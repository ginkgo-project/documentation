# FixedCoarsening: user-specified coarse rows

`gko::multigrid::FixedCoarsening` is a deterministic coarsening method where the
caller picks which fine-level rows survive into the coarse level. The selection is purely the caller's: no aggregation, no matching, no
value-based weighting happens.
This makes it a niche choice, but the right one in two scenarios:

1. **Hierarchical-mesh problems.** When your discretisation already provides a
   natural hierarchy of unknowns (e.g., a uniform refinement of a finite-element
   mesh), the coarse-row indices are known ahead of time. FixedCoarsening lets you
   pass them in directly without going through an algebraic coarsening that would
   recompute the hierarchy.
2. **Reproducible test cases.** Because the coarse rows are fixed, the resulting
   multigrid hierarchy is bitwise reproducible across runs and platforms, useful
   for regression testing and for isolating multigrid issues from coarsening
   issues.

## How it works

The user supplies a `gko::array<IndexType>` listing the global indices (in the fine
matrix) of rows to keep. FixedCoarsening builds:

- A restriction operator that selects the `coarse_rows` entries from a fine-level
  vector.
- A prolongation operator (the transpose of the restriction) that injects coarse
  values back into the fine-level positions.
- A coarse operator `A_{i+1} = R A_i P`, equivalent to the principal sub-matrix of
  `A_i` indexed by `coarse_rows`.

The prolongation is pure injection: entries at non-selected fine-level positions
are zero in the prolonged correction. This is intentional and what makes
FixedCoarsening "fixed": the prolongation operator is not smoothed and carries no
interpolation weights.

## Factory parameters

| Parameter | Type | Purpose |
|---|---|---|
| `coarse_rows` | `array<IndexType>` | The required input: global row indices of the fine matrix to keep as coarse rows. Must be non-empty (debug builds assert on a missing or empty array). Default `nullptr`. |
| `skip_sorting` | `bool` | If you can guarantee that the input system matrix is sorted (first by row, then by column index), set `true` to skip the internal sort. Wrong-setting silently corrupts the coarsening; see [`csr.md`'s sorted-column-indices section](../../formats/csr.md#sorted-column-indices) for context on when sortedness can or cannot be assumed. Default `false`. |

## When to use FixedCoarsening

Choose FixedCoarsening when:
- You have explicit knowledge of which DOFs should appear at each coarse level,
  typically from a geometric or hierarchical mesh.
- You want reproducibility above performance: tests, debugging, didactic uses.
- You are experimenting with custom coarsening strategies and want a
  deterministic baseline.

Avoid FixedCoarsening when:
- You are setting up multigrid on a generic matrix without prior knowledge of the
  unknown structure: [Pgm](pgm.md), [Pmis](pmis.md) and [Rs](rs.md) choose the coarse
  level algebraically from the matrix values, and
  [UniformCoarsening](uniform-coarsening.md) picks it by a constant index stride.
- The problem requires interpolation (smoothed prolongation) for accuracy;
  FixedCoarsening's pure-injection prolongation is too coarse for many AMG
  use-cases.

## Construction example

```cpp
gko::array<gko::int32> coarse_indices(exec, {0, 2, 4, 6, 8, 10});

auto fixed = gko::share(
    gko::multigrid::FixedCoarsening<double, gko::int32>::build()
        .with_coarse_rows(std::move(coarse_indices))
        .on(exec));

auto mg = gko::solver::Multigrid::build()
              .with_mg_level(fixed)
              .with_max_levels(1u)   // typically only one level for fixed coarsening
              .with_criteria(gko::stop::max_iters(1u))
              .on(exec);
```

Because FixedCoarsening produces only one level (it expects user-specified rows for
that level), `max_levels` is usually set to `1`. If you need multiple levels of
fixed coarsening, supply multiple `mg_level` factories and use a `level_selector` to
pick the right indices per level.

:::{seealso}
- [Multigrid](../multigrid.md): how FixedCoarsening fits into the overall hierarchy.
- [Pgm](pgm.md), [Pmis](pmis.md), [Rs](rs.md), [UniformCoarsening](uniform-coarsening.md): the automatic alternatives for generic matrices.
- API reference: [`gko::multigrid::FixedCoarsening`](../../../../api-reference/multigrid/fixed-coarsening.md)
:::
