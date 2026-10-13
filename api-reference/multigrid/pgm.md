# `gko::multigrid::Pgm`

Parallel Graph Matching aggregation — Ginkgo's default algebraic
coarsening. Pgm builds the coarse level by aggregating fine-level rows
in two phases:

1. For every unaggregated row $i$, find the *strongest neighbour* —
   the column $j$ in $i$'s row with the largest symmetric weight
   $(|a_{ij}| + |a_{ji}|)/2$, scaled by the larger of the diagonal
   entries $|a_{ii}|$ and $|a_{jj}|$.
2. Match rows whose strongest-neighbour relation is mutual (row $i$
   picks $j$ and row $j$ picks $i$), forming aggregates of size 2.

The handshaking repeats until `max_iterations` passes (default 15) have
run or the unaggregated fraction of rows falls below
`max_unassigned_ratio` (default 0.05); the remaining rows are attached
to an existing aggregate or left as singletons. The current implementation supports
aggregates of size 2 (one-phase handshaking); the resulting coarse
operator is built by Galerkin projection $A_c = R\, A\, P$.

Pgm is the natural default coarsening for non-geometric problems
where no separator information is available — graph matching captures
the strongest couplings using the matrix values themselves.

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
