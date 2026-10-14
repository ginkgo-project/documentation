# `gko::multigrid::Pgm`

Parallel Graph Matching aggregation: Ginkgo's default algebraic
coarsening. Pgm builds the coarse level by aggregating fine-level rows
in two steps:

1. For every unaggregated row $i$, find the *strongest neighbour*:
   the column $j$ with the largest symmetric weight $(|a_{ij}| + |a_{ji}|)/2$,
   divided by the larger of the diagonal entries $|a_{ii}|$ and $|a_{jj}|$.
2. Match rows whose strongest-neighbour relation is mutual (row $i$
   picks $j$ and row $j$ picks $i$), forming aggregates of size 2.

Steps 1 and 2 are repeated until `max_iterations` passes (default 15) have
run, a pass makes no new match, or the number of unaggregated rows drops below
`max_unassigned_ratio` (default 0.05) times the number of rows; the remaining rows are then attached to a neighbouring
aggregate or left as singletons. The current implementation forms aggregates
of size 2 (one-phase handshaking); the coarse operator is built by Galerkin
projection $A_c = R\, A\, P$.

The `Pgm` factory supports `generate_reuse()`: once a matrix has been recorded,
later matrices with the same sparsity pattern keep its aggregates and only the
coarse matrix is recomputed. The aggregates come from the values of the first
matrix, so it should be representative of the later ones.

:::{attention}
The reuse interface is experimental and may change.
:::

Pgm is the natural default coarsening for non-geometric problems
where no separator information is available; graph matching captures
the strongest couplings using the matrix values themselves. Pgm accepts
`gko::experimental::distributed::Matrix` as well as `Csr` matrices.

```{doxygenclass} gko::multigrid::Pgm
:members:
```

:::{seealso}
Reference:
Naumov, M., Arsaev, M., Castonguay, P., Cohen, J., Demouth, J.,
Eaton, J., Layton, S., Markovskiy, N., Reguly, I., Sakharnykh, N.,
Sellappan, V., Strzodka, R. *AmgX: A Library for GPU Accelerated
Algebraic Multigrid and Preconditioned Iterative Methods.*
SIAM Journal on Scientific Computing, 37 (5), S602–S626, 2015.
<https://doi.org/10.1137/140980260>
:::
