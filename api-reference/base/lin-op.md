# `gko::LinOp`

Abstract base for every linear operator in Ginkgo. A `LinOp` exposes
the action $x \leftarrow A b$ through `apply(b, x)` (and the scaled
form $x \leftarrow \alpha A b + \beta x$ through the four-argument
overload). Everything that behaves like an operator — sparse and dense
matrices, factorisations, Krylov solvers, preconditioners, composed
operators — is a `LinOp`, so callers can treat them uniformly: feed any
of them to a Krylov solver as the system matrix, the preconditioner,
or both.

Concrete classes derive from `LinOp` directly and implement
`apply_impl`. `LinOp::apply` logs the call, validates the operand
sizes, makes temporary copies of operands that live on a different
executor, and then calls `apply_impl`.

```{doxygenclass} gko::LinOp
:members:
```
