# `gko::preconditioner::GaussSeidel`

Gauss-Seidel preconditioner: the $\omega = 1$ special case of SOR.
Applies one forward (or symmetric) sweep over the system matrix as the
approximate inverse. Cheap and embarrassingly simple, but with
convergence that depends strongly on the matrix structure.

```{doxygenclass} gko::preconditioner::GaussSeidel
:members:
```
