# `gko::stop::ResidualNormReduction`

Stops when the residual norm has been reduced by a given factor
relative to the initial residual: $\|r_k\|_2 \le \mathrm{reduction} \cdot
\|r_0\|_2$. Functionally equivalent to `RelativeResidualNorm`; prefer
that name in new code — this class is kept for backwards compatibility.

```{doxygenclass} gko::stop::ResidualNormReduction
:members:
```
