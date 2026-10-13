# `gko::experimental::solver::`<br>`Direct`

Direct solver built from a sparse $LU$ or Cholesky factorisation
supplied by a [`Factorization`](../factorization/factorization.md)
factory. The factorisation is computed once during `generate()` and
reused across every `apply()`; each apply runs a forward solve on
the lower factor ($L$) followed by a backward solve on the upper factor ($U$ for $LU$, $L^H$ for
Cholesky).

The factorisation factory is provided as the `factorization` factory
parameter — typically a built-in factorisation
([`Cholesky`](../factorization/cholesky.md),
[`Lu`](../factorization/lu.md)) or a user-supplied one. The native
Ginkgo triangular solvers
([`LowerTrs`](lower-trs.md), [`UpperTrs`](upper-trs.md)) handle the
substitutions.

The `num_rhs` factory parameter (default 1) states the number of right-hand
sides; the CUDA executor throws if `apply` receives a different number, so set
it to the width of your right-hand-side block when you run on CUDA.

For an introduction to direct solvers in Ginkgo, see the
[direct solver concept page](../../user-guide/concepts/solvers/direct.md).

```{doxygenclass} gko::experimental::solver::Direct
:members:
```
