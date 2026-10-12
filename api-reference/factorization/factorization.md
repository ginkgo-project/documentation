# `gko::experimental::factorization::`<br>`Factorization`

The container produced by every factorisation in this namespace. Owns
the computed factors and dispatches on `apply`: depending on the
internal representation, applying the `Factorization` performs the
appropriate combination of triangular solves and matrix multiplies that
realises the action of $A^{-1}$ (or its approximation).

:::{attention}
This class lives in the `experimental` namespace; its interface may change.
:::

```{doxygenclass} gko::experimental::factorization::Factorization
:members:
```
