# `gko::multigrid::FixedCoarsening`

The simplest possible level generator: select coarse-level rows by a
user-supplied index list. Restriction selects those rows and
prolongation injects by transposition ($R$ holds the selected rows of
the identity, $P = R^T$). No interpolation, no weighting — coarse value
$k$ equals fine value at index $i_k$, and prolongation places it back.

Use `FixedCoarsening` when the application already knows which rows
should remain on the coarse level. Typical sources of such knowledge:

- **Geometric coarsening** on a structured mesh, where every other
  point in each dimension is a natural coarse-level node.
- **Domain decomposition** where the interface DOFs should remain on
  the coarse problem.
- **Custom solver schemes** that benchmark a fixed-coarsening baseline
  against algebraic alternatives like [`Pgm`](pgm.md).

Construction takes the index array of selected fine-level rows; the
coarse operator is built by Galerkin projection $A_c = R A P$ from
those selectors.

```{doxygenclass} gko::multigrid::FixedCoarsening
:members:
```
