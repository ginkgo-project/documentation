# `gko::matrix::Sellp`

Sliced ELL. Partitions rows into fixed-size slices (`slice_size`,
default 64) and pads each slice independently to its own longest row,
with `stride_factor` controlling the padding granularity. A middle
ground between ELL's regular layout and CSR's storage efficiency.

```{doxygenclass} gko::matrix::Sellp
:members:
```
