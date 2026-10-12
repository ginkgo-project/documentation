# UniformCoarsening — constant-stride coarsening

`gko::multigrid::UniformCoarsening` selects coarse rows by a constant stride over the
row index space. Neither the matrix values nor any mesh geometry are consulted, which
makes the setup cheap and the hierarchy predictable. It suits problems whose unknowns
are numbered along a regular structure, and it serves as a baseline for comparing
other coarsenings.

## How it works

With `coarse_skip = s`, the coarse level has `ceil(n / s)` rows. The `aggregation`
parameter decides how fine rows take part:

- **Aggregation** (`aggregation = true`): fine row `i` is mapped to coarse row
  `floor(i / s)`. The restriction sums each block of `s` consecutive fine rows and
  the prolongation copies the coarse value back to them, so the Galerkin coarse matrix
  `A_{i+1} = R A_i P` keeps the connectivity of the fine graph.
- **Injection** (`aggregation = false`): only the rows with `i % s == 0` take part.
  The coarse graph can become disconnected.

## Factory parameters

| Parameter | Type | Purpose |
|---|---|---|
| `coarse_skip` | `int` | The stride between coarse rows, and the number of fine rows per coarse row. Default `2`. |
| `aggregation` | `bool` | `true` selects aggregation-style coarsening, `false` injection-style coarsening. Default `true`. |
| `skip_sorting` | `bool` | Set to `true` when the input is a `Csr` matrix sorted by row and then by column index, to skip the internal sort. Default `false`. |

## When to use it

UniformCoarsening is a good choice when:
- the unknowns are ordered along a regular structure, so that neighbouring indices are
  strongly coupled (structured grids, banded systems);
- you need a very cheap setup, or a hierarchy for distributed matrices that is cheap to set up;
- you want a simple baseline against which to compare [Pgm](pgm.md) or the classical
  coarsenings.

It is not a good choice for matrices with an irregular numbering, where index
neighbours say nothing about the coupling. Use an algebraic coarsening there.

## Distributed matrices

UniformCoarsening accepts `gko::experimental::distributed::Matrix`, and so does
[Pgm](pgm.md). For a distributed matrix each rank coarsens its own rows, and the
coarse operator, restriction and prolongation are distributed matrices on the same
communicator, so `Multigrid` continues with distributed levels. This path is available
when Ginkgo is built with `-DGINKGO_BUILD_MPI=ON`.

## Construction example

```cpp
auto mg = gko::solver::Multigrid::build()
              .with_mg_level(gko::multigrid::UniformCoarsening<double, int>::build()
                                 .with_coarse_skip(2))
              .with_criteria(gko::stop::max_iters(1u))
              .on(exec);
```

The `UniformCoarsening` builder is handed to `with_mg_level` directly. `Multigrid`
requires `with_criteria`, even when it is used as a preconditioner.

:::{seealso}
- [Multigrid](../multigrid.md) — how UniformCoarsening fits into the overall hierarchy.
- [Pgm](pgm.md) — the value-based aggregation that also supports distributed matrices.
- [FixedCoarsening](fixed-coarsening.md) — coarse rows chosen by the caller.
- API reference: [`gko::multigrid::UniformCoarsening`](../../../../api-reference/multigrid/uniform-coarsening.md)
:::
