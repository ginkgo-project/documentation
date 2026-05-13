# Pgm — parallel graph match

`gko::multigrid::Pgm` is Ginkgo's aggregation-based AMG coarsening. It implements the
parallel graph matching algorithm from NVIDIA's AmgX paper (Naumov et al., 2015) and
is the recommended starting point for algebraic multigrid on most problems.

## How it works

Pgm groups fine-level rows into aggregates, where each aggregate becomes a single
row in the coarse matrix. The current implementation produces aggregates of size two
(one-phase handshaking) and uses **values, not just structure**, to decide which
neighbour to match each row with — strong coupling makes a stronger pairing
candidate.

The procedure for each iteration:

1. **Strongest-neighbour selection.** For each unaggregated row `i`, identify the
   neighbour `j` reachable via the largest absolute matrix entry.
2. **Mutual-match aggregation.** If row `j`'s strongest neighbour is also `i`, merge
   the pair into a new aggregate.
3. **Repeat** until either every row is aggregated or the
   `max_unassigned_ratio` is reached.

Rows still unaggregated after `max_iterations` are folded into an existing
neighbouring aggregate (or kept singleton if no neighbour is available).

The result is a row-to-aggregate mapping (`get_const_agg()`) that drives construction
of the prolongation `P`, the restriction `R = P^T`, and the coarse operator
`A_{i+1} = R A_i P`. These four operators satisfy the `MultigridLevel` contract.

## Factory parameters

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `max_iterations` | `unsigned` | `15` | Upper bound on the number of handshaking iterations. The default mirrors the NVIDIA AMGX reference manual. |
| `max_unassigned_ratio` | `double` | `0.05` | Stop iterating when at most this fraction of rows remain unaggregated. Lower values demand more complete aggregation; higher values trade aggregation completeness for fewer iterations. |
| `deterministic` | `bool` | `false` | When `true`, the matching is reproducible across runs (at extra cost). When `false`, the order of strong-neighbour resolution can depend on parallel scheduling — different runs may produce different aggregations of the same matrix, though always valid ones. Use `true` for debugging and regression testing. |
| `skip_sorting` | `bool` | `false` | If you can guarantee that the input CSR is sorted by column index within each row, set `true` to skip Pgm's internal sort. Wrong-set value silently produces incorrect aggregations. |

## When to use Pgm

Pgm is the right starting choice when:
- You are setting up `Multigrid` for the first time on a new problem class.
- The matrix is symmetric or near-symmetric — the value-based matching assumes the
  off-diagonal magnitudes are meaningful.
- You have no special knowledge of the unknown ordering or geometry.

Pgm may underperform when:
- The matrix is strongly non-symmetric — value-based matching can pair unrelated
  rows.
- The problem has no clear notion of "strong coupling" between unknowns (purely
  combinatorial problems, for instance).
- Reproducibility across runs matters but performance does not — set
  `deterministic = true`, but expect a slowdown.

## Construction example

```cpp
auto pgm = gko::multigrid::Pgm<double, gko::int32>::build()
               .with_max_iterations(15u)
               .with_max_unassigned_ratio(0.05)
               .with_deterministic(false)
               .on(exec);

auto mg = gko::solver::Multigrid::build()
              .with_mg_level(pgm)
              .with_max_levels(10u)
              .on(exec);
```

The `Pgm` factory is passed to `Multigrid` via `with_mg_level`. `Multigrid` then
calls `pgm->generate(A_i)` once per level during its own `generate()` step,
producing the level hierarchy.

## Distributed Pgm

Pgm is the only coarsening method in Ginkgo that supports distributed matrices —
`gko::experimental::distributed::Matrix<V, L, G>`. The same `Pgm` factory accepts
both single-rank `gko::matrix::Csr` and the distributed matrix type; `generate()`
detects which is which at runtime and takes one of two branches
(see the distributed branch of `Pgm::generate()` in `core/multigrid/pgm.cpp`).

### How the distributed path works

When the input is a distributed matrix, Pgm runs the aggregation per rank and
stitches the results together with a single all-to-all communication. The procedure:

1. **Local aggregation.** Each rank runs the standard one-phase handshaking
   algorithm on the *diagonal matrix* (`get_diag_matrix()`) of its share of the
   matrix. Off-diagonal entries — couplings to rows owned by other ranks — are
   ignored at this step. Each rank ends up with a local-aggregate index per local
   row.
2. **Coarse-partition setup.** Pgm constructs a block-wise coarse partition where
   each rank owns a contiguous range of coarse-row global indices, sized by that
   rank's local aggregate count. The result is one range per part, one part per
   rank.
3. **Cross-rank aggregate exchange.** For every off-diagonal column index that this
   rank reads from (and that another rank owns), the other rank's aggregate ID for
   that row is needed in order to translate the off-diagonal matrix into coarse-row
   indices. Pgm uses the matrix's row-gatherer to issue an `i_all_to_all_v` so each
   rank receives the coarse global aggregate IDs for the off-diagonal columns it
   touches (see `communicate_off_diag_agg` in `core/multigrid/pgm.cpp`). When the executor
   does not support direct GPU-side MPI, the buffer is staged through host memory.
4. **Coarse index map and matrix assembly.** Pgm builds a coarse-level
   `experimental::distributed::index_map` from the received off-diagonal aggregate
   IDs, then constructs three new `distributed::Matrix`es: the coarse operator, the
   restriction, and the prolongation. Each carries the same communicator as the
   input matrix.

The communication cost per coarsening step is one `i_all_to_all_v` whose payload
size is the total number of off-diagonal column connections — i.e. it scales with
the existing communication volume of the matrix, not with the matrix size.

### What this means for the hierarchy

- The coarse matrix is itself a `distributed::Matrix`, so Multigrid's recursive
  coarsening continues to use distributed Pgm at every level. There is no automatic
  consolidation onto a single rank as the matrix shrinks — if you want the
  coarsest level to live on one process (often the right choice once the global
  size is small), gather it explicitly before configuring Multigrid's
  `coarsest_solver`.
- The aggregation is purely local at the row-aggregate-decision level. Couplings
  across ranks affect only the coarse operator's off-diagonal matrix, not which
  rows get grouped together. This means the quality of the coarsening can degrade
  near rank boundaries on poorly-partitioned matrices — apply a graph partitioner
  first if rank boundaries are arbitrary.
- The two value-vs-structure parameters (`max_iterations`,
  `max_unassigned_ratio`) and the `deterministic` parameter have the same meaning
  as in the single-rank case. They control only the local handshaking step, not
  the cross-rank exchange.

### Required configuration

Distributed Pgm requires Ginkgo to have been built with
`-DGINKGO_BUILD_MPI=ON`. The Pgm header guards the distributed code path with
`#if GINKGO_BUILD_MPI`; without MPI support, Pgm only accepts single-rank
matrices.

## Publications

- [Three-Precision Algebraic Multigrid on GPUs](https://doi.org/10.1016/j.future.2023.07.024)
- [Mixed Precision Algebraic Multigrid on GPUs](https://doi.org/10.1007/978-3-031-30442-2_9)

:::{seealso}
- [Multigrid](../multigrid.md) — how Pgm fits into the overall hierarchy.
- [FixedCoarsening](fixed-coarsening.md) — the alternative coarsening method.
- API reference: [`gko::multigrid::Pgm`](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1multigrid_1_1Pgm.html)
:::
