# `gko::matrix::Fbcsr`

Fixed-size block CSR. Stores nonzeros as dense `block_size × block_size`
sub-blocks indexed by CSR-style block-row pointers and block-column
indices. The right choice for matrices arising from vector PDEs or
finite-element systems with a small, fixed number of unknowns per
mesh entity.

```{doxygenclass} gko::matrix::Fbcsr
:members:
```
