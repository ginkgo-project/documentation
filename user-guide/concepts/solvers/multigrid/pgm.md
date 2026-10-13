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
   neighbour `j` with the strongest symmetric coupling: the average of `|a_ij|` and
   `|a_ji|` relative to the larger of the two diagonal entries.
2. **Mutual-match aggregation.** If row `j`'s strongest neighbour is also `i`, merge
   the pair into a new aggregate.
3. **Repeat** until a pass makes no new match, the number of unaggregated rows
   drops below `max_unassigned_ratio` times the number of rows, or
   `max_iterations` passes have run.

Rows still unaggregated after that are folded into an existing
neighbouring aggregate (or kept singleton if no neighbour is available).

The result is a row-to-aggregate mapping (`get_const_agg()`) that drives construction
of the prolongation `P`, the restriction `R = P^T`, and the coarse operator
`A_{i+1} = R A_i P`. These four operators satisfy the `MultigridLevel` contract.

### Reusing aggregates

The `Pgm` factory supports `generate_reuse()`. Once a matrix has been recorded, later
matrices with the same sparsity pattern keep its aggregates and only the coarse
matrix is recomputed, so the result can differ from a fresh `generate()`. The
aggregates come from the values of the first matrix, so it should be representative
of the later ones. `Multigrid` uses this when its own `generate_reuse()` is called;
see [Reusing the hierarchy](../multigrid.md#reusing-the-hierarchy).

:::{attention}
The reuse interface is experimental and may change.
:::

## Factory parameters

| Parameter | Type | Purpose |
|---|---|---|
| `max_iterations` | `unsigned` | Upper bound on the number of handshaking iterations. Default `15`, which mirrors the NVIDIA AMGX reference manual. |
| `max_unassigned_ratio` | `double` | Stop iterating when fewer than this fraction of the rows remain unaggregated. Lower values demand more complete aggregation; higher values trade aggregation completeness for fewer iterations. Default `0.05`. |
| `deterministic` | `bool` | `true` selects the deterministic `assign_to_exist_agg` step, so the same matrix always yields the same aggregates (at extra cost). `false` lets that step depend on the execution order, so different runs may produce different, always valid, aggregations of the same matrix. Use `true` for debugging and regression testing. Default `false`. |
| `skip_sorting` | `bool` | If you can guarantee that the input CSR is sorted by column index within each row, set `true` to skip Pgm's internal sort. Wrong-set value silently produces incorrect aggregations. Default `false`. |

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
auto pgm = gko::share(gko::multigrid::Pgm<double, gko::int32>::build()
                          .with_max_iterations(15u)
                          .with_max_unassigned_ratio(0.05)
                          .with_deterministic(false)
                          .on(exec));

auto mg = gko::solver::Multigrid::build()
              .with_mg_level(pgm)
              .with_max_levels(10u)
              .with_criteria(gko::stop::max_iters(1u))
              .on(exec);
```

The `Pgm` factory is passed to `Multigrid` via `with_mg_level`; `.on(exec)` returns a
`std::unique_ptr`, so it is wrapped in `gko::share` first. `Multigrid` then
calls `pgm->generate(A_i)` once per level during its own `generate()` step,
producing the level hierarchy.

## Distributed Pgm

Pgm and [UniformCoarsening](uniform-coarsening.md) are the coarsening methods that
accept distributed matrices —
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
  coarsening continues to use distributed Pgm at every level, with every level spread
  over all ranks. To place the coarsest level on one process (often the right choice once the global
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

:::{seealso}
- [Multigrid](../multigrid.md) — how Pgm fits into the overall hierarchy.
- [Pmis](pmis.md), [Rs](rs.md), [UniformCoarsening](uniform-coarsening.md) — the other automatic coarsenings.
- [FixedCoarsening](fixed-coarsening.md) — coarse rows chosen by the caller.
- Publications: [Three-Precision Algebraic Multigrid on GPUs](https://doi.org/10.1016/j.future.2023.07.024), [Mixed Precision Algebraic Multigrid on GPUs](https://doi.org/10.1007/978-3-031-30442-2_9)
- API reference: [`gko::multigrid::Pgm`](../../../../api-reference/multigrid/pgm.md)
:::
