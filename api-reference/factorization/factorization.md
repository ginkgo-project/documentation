# `gko::experimental::factorization::`<br>`Factorization`

The container produced by every factorisation in this namespace. Owns
the computed factors, stored in one of several representations, and hands
them to [`gko::experimental::solver::Direct`](../solver/direct.md), which
uses them for the triangular solves. When the factors are kept as a
composition, `apply` multiplies by the factors.

Lives in the `experimental` namespace pending API stabilisation.

```{doxygenclass} gko::experimental::factorization::Factorization
:members:
```
