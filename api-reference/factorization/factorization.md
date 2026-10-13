# `gko::experimental::factorization::`<br>`Factorization`

The container produced by every factorisation in this namespace. Owns
the computed factors, stored in one of several representations, and hands
them to `gko::experimental::solver::Direct`, which uses them for the
triangular solves. When the factors are kept as a composition, `apply`
multiplies by the factors.

:::{attention}
This class lives in the `experimental` namespace; its interface may change.
:::

```{doxygenclass} gko::experimental::factorization::Factorization
:members:
```
