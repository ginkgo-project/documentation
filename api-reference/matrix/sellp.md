# `gko::matrix::Sellp`

Sliced ELL with padding (SELL-P). Partitions rows into fixed-size slices
(`slice_size`) and pads each slice independently to its own longest row,
rounded up to a multiple of `stride_factor`. A middle ground between
ELL's regular layout and CSR's storage efficiency.

```{doxygenclass} gko::matrix::Sellp
:members:
```
