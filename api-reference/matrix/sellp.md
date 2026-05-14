# `gko::matrix::Sellp`

Sliced ELL with permutation. Partitions rows into fixed-size slices,
pads each slice independently to its own longest row, and optionally
permutes rows within a slice to balance work. A middle ground between
ELL's regular layout and CSR's storage efficiency.

```{doxygenclass} gko::matrix::Sellp
:members:
```
