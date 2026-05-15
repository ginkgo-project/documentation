# `gko::factorization::Ic`

IC(0): incomplete Cholesky factorisation with the same sparsity
pattern as the input matrix. Stores a lower-triangular factor $L$
together with its conjugate transpose $L^H$ that approximate
$A \approx L L^H$ on $\mathcal{S}(A)$.

```{doxygenclass} gko::factorization::Ic
:members:
```
