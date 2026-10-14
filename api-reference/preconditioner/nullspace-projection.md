# `gko::preconditioner::NullspaceProjection`

Wraps a preconditioner $M$ with projections onto the complement of a nullspace,
computing $z = P M Q r$. It applies $M$, removes the right nullspace ($P$) from the
result and, if a left nullspace is given, removes it ($Q$) from the input first.

`Cg`, `Fcg`, `Minres`, `Gmres` and `Bicgstab` create this wrapper around their
preconditioner automatically when a `nullspace` is set. It can also be built by hand and used as the
preconditioner of any solver (with a solver that does not handle the nullspace itself, project `b` and the initial guess by hand with `Nullspace::project`); the factory parameters are `preconditioner`,
`generated_preconditioner`, `nullspace` and `left_nullspace`. The system matrix
has to be square.

```{doxygenclass} gko::preconditioner::NullspaceProjection
:members:
```

:::{seealso}
- [Singular systems and nullspaces](../../user-guide/concepts/solvers/singular-systems.md): concept page.
- [`gko::solver::Nullspace`](../solver/nullspace.md): the projector passed as `nullspace`.
:::
