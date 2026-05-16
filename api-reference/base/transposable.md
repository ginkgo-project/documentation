# `gko::Transposable`

Trait interface for operators that can produce their transpose and
conjugate transpose. Provides two virtual methods, `transpose()` and
`conj_transpose()`, both returning a fresh `unique_ptr<LinOp>` that
represents the transformed operator. Concrete classes that implement
this trait include `matrix::Csr`, `matrix::Dense`, the Krylov solvers
(which return the solver for the transposed system), and the
preconditioners that support it.

Use `gko::as<Transposable>(linop)` to access the trait at runtime when
working with a generic `LinOp` pointer.

```{doxygenclass} gko::Transposable
:members:
```
