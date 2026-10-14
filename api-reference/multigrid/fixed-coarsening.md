# `gko::multigrid::FixedCoarsening`

The simplest possible level generator: select coarse-level rows by a
user-supplied index list, then build the level from that
selection. No interpolation, no weighting; restriction selects those rows and
prolongation injects by transposition ($R$ holds the selected rows of the
identity, $P = R^T$).

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
