# Rs: Ruge–Stüben coarsening

`gko::multigrid::Rs` is the classical Ruge–Stüben algebraic-multigrid coarsening for
M-matrices: matrices with a strictly positive diagonal and non-positive off-diagonal
entries, as produced by many diffusion discretisations. It produces a C/F splitting,
an interpolation operator `P` and the Galerkin coarse operator.

## How it works

A level is built in three steps:

1. **Strength of connection.** Column `j` strongly influences row `i` when
   `-a_ij >= θ * max_{k != i} (-a_ik)`, where `θ` is the `strength_threshold`.
2. **Greedy C/F selection.** Repeatedly, the undecided row with the most strong
   neighbours becomes a C point, its undecided strong neighbours become F points, and
   the neighbour counts are updated. The selection is inherently sequential, so it runs
   on the host: on a device executor the matrix data is copied to the host for this
   step and the result is copied back.
3. **Classical interpolation.** F points are interpolated from their strong C
   neighbours with the classical Ruge–Stüben direct formula, which also accounts for
   strong F neighbours. The restriction is `R = P^T` and the coarse operator is
   `A_{i+1} = R A_i P`.

## Factory parameters

| Parameter | Type | Purpose |
|---|---|---|
| `strength_threshold` | `double` | The threshold `θ` of the strength-of-connection test. Default `0.25`. |
| `skip_sorting` | `bool` | Set to `true` when the input is a `Csr` matrix sorted by row and then by column index, to skip the internal sort. Default `false`. |
| `skip_m_matrix_check` | `bool` | Set to `true` to skip the check that the matrix is an M-matrix. The check can be expensive on large matrices. Default `false`. |

## When to use it

Rs is a good choice when:
- the matrix is an M-matrix, such as a standard finite-difference or finite-element
  discretisation of the Poisson equation;
- convergence per cycle matters more than setup time;
- the hierarchy is built once and applied many times.

Rs coarsens non-distributed matrices. For distributed matrices use
[Pgm](pgm.md) or [UniformCoarsening](uniform-coarsening.md).

:::{attention}
Generation throws `gko::NotSupported` when the matrix is not an M-matrix, unless
`skip_m_matrix_check` is `true`. Because the C/F selection runs on the host, the setup on
GPUs is usually slower than with [Pmis](pmis.md).
:::

## Construction example

```cpp
auto rs = gko::share(gko::multigrid::Rs<double, int>::build()
                         .with_strength_threshold(0.25)
                         .on(exec));
auto mg = gko::solver::Multigrid::build()
              .with_mg_level(rs)
              .with_max_levels(10u)
              .with_criteria(gko::stop::max_iters(1u))
              .on(exec);
```

A factory created with `.on(exec)` is a `std::unique_ptr`, so it is wrapped in
`gko::share` before it is passed to `with_mg_level` (passing the builder without
`.on(exec)` works as well). `Multigrid` requires `with_criteria`.

:::{seealso}
- [Multigrid](../multigrid.md): how Rs fits into the overall hierarchy.
- [Pmis](pmis.md): the parallel classical alternative.
- API reference: [`gko::multigrid::Rs`](../../../../api-reference/multigrid/rs.md)
- J. W. Ruge, K. Stüben, [Algebraic multigrid](https://doi.org/10.1137/1.9781611971057.ch4), in Multigrid Methods, SIAM, 1987.
:::
