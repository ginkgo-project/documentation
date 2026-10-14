# Pmis: parallel modified independent set

`gko::multigrid::Pmis` is a classical algebraic-multigrid coarsening. Instead of
merging rows into aggregates, it splits the unknowns into *coarse* (C) points that
survive on the next level and *fine* (F) points that are interpolated from them. The
splitting is chosen by a parallel modified independent set algorithm, which is why it
maps well onto GPUs.

## How it works

A level is built in three steps:

1. **Strength of connection.** Row `i` depends strongly on column `j` when
   `|a_ij| >= θ * max_{k != i} |a_ik|`, where `θ` is the `strength_threshold`.
2. **C/F splitting.** Every row gets a weight, the number of rows that depend
   strongly on it plus a pseudo-random number in `[0, 1)`. In each pass, rows whose
   weight is a local maximum among their strong neighbours become C points, and the
   rows that depend strongly on a new C point become F points. The passes repeat until
   every row is classified.
3. **Direct interpolation.** An F point is interpolated from its strongly connected
   C neighbours, a C point is injected. This gives the prolongation `P`; the
   restriction is `R = P^T` and the coarse operator is `A_{i+1} = R A_i P`.

Each row's decision in a pass depends only on the classification from the previous
pass, so the rows of a pass can be processed independently of each other. This is what
makes Pmis suitable for GPUs.

:::{attention}
The pseudo-random weights make the C/F splitting depend on the executor. The coarse
grid computed on a CPU executor can therefore differ from the one computed on a GPU
executor, and so can the resulting convergence history.
:::

## Factory parameters

| Parameter | Type | Purpose |
|---|---|---|
| `strength_threshold` | `remove_complex<ValueType>` | The threshold `θ` of the strength-of-connection test, usually chosen between 0.25 and 0.5. Default `0.25`. |
| `skip_sorting` | `bool` | Set to `true` when the input matrix is a `Csr` matrix sorted by row and then by column index, to skip the internal sort. An unsorted matrix with this flag set gives an incorrect level. Default `false`. |

## When to use it

Pmis is a good choice when:
- you want classical C/F coarsening with interpolation instead of aggregation;
- setup time matters on GPUs, since the splitting needs no sequential pass;
- the matrix is a scalar problem with a meaningful sign pattern, such as a diffusion
  or Poisson discretisation.

Pmis coarsens non-distributed matrices. For distributed matrices use
[Pgm](pgm.md) or [UniformCoarsening](uniform-coarsening.md).

:::{attention}
Generation throws `gko::InvalidStateError` when the splitting selects no C point, so
that the coarse level would be empty. Set `min_coarse_rows` or `max_levels` on
`Multigrid` so that very small matrices are not coarsened further.
:::

## Construction example

```cpp
auto solver =
    gko::solver::Cg<double>::build()
        .with_preconditioner(
            gko::solver::Multigrid::build()
                .with_mg_level(gko::multigrid::Pmis<double, int>::build()
                                   .with_strength_threshold(0.25))
                .with_criteria(gko::stop::max_iters(1u)))
        .with_criteria(gko::stop::max_iters(1000u),
                       gko::stop::relative_residual_norm(1e-8))
        .on(exec)
        ->generate(A);
solver->apply(b, x);
```

The `Pmis` builder is handed to `with_mg_level` directly, and `Multigrid` calls its
`generate()` once per level. `Multigrid` requires `with_criteria`, even when it is
used as a preconditioner; `max_iters(1u)` makes it apply one cycle per outer
iteration.

:::{seealso}
- [Multigrid](../multigrid.md): how Pmis fits into the overall hierarchy.
- [Rs](rs.md): the sequential classical alternative for M-matrices.
- API reference: [`gko::multigrid::Pmis`](../../../../api-reference/multigrid/pmis.md)
- H. De Sterck, U. M. Yang, J. J. Heys, [Reducing complexity in parallel algebraic multigrid preconditioners](https://doi.org/10.1137/040615729), SIAM J. Matrix Anal. Appl. 27 (4), 2006.
:::
