# `gko::factorization::Ilu`

ILU(0): incomplete LU factorisation with the same sparsity pattern as
the input matrix. Stores a lower unitriangular factor $L$ and an upper
triangular factor $U$ that approximate $A \approx L U$ on $\mathcal{S}(A)$.

```{doxygenclass} gko::factorization::Ilu
:members:
```
