# `gko::LinOp`

Abstract base for every linear operator in Ginkgo. A `LinOp` exposes
the action $x \leftarrow A b$ through `apply(b, x)` (and the scaled
form $x \leftarrow \alpha A b + \beta x$ through the four-argument
overload). Everything that behaves like an operator — sparse and dense
matrices, factorisations, Krylov solvers, preconditioners, composed
operators — is a `LinOp`, so callers can treat them uniformly: feed any
of them to a Krylov solver as the system matrix, the preconditioner,
or both.

Concrete classes do not derive from `LinOp` directly. They use the
CRTP helper [`EnableLinOp`](enable-linop.md), which supplies the
default copy/clone, the apply dispatcher, and the polymorphic-object
plumbing so the implementation only writes `apply_impl`.

```{doxygenclass} gko::LinOp
:members:
```
