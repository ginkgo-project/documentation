# `gko::solver::Nullspace`

Describes the right or left nullspace of an operator and acts as the orthogonal
projector $P = I - V V^H$ onto its complement, where $V$ is an orthonormal basis
of the nullspace. A `Nullspace` is created from explicit basis vectors, from the
constant vector, or from both, and is passed as the `nullspace` or
`left_nullspace` factory parameter of `Cg`, `Fcg`, `Minres`, `Gmres` or
`Bicgstab` to solve singular systems.

The explicit basis is copied and orthonormalized once at creation, and
numerically dependent columns are dropped. A constant-only nullspace created with
`create_from_constant` is never stored and adapts to any matrix size, value type
and partition.

```{doxygenclass} gko::solver::Nullspace
:members:
```

:::{seealso}
- [Singular systems and nullspaces](../../user-guide/concepts/solvers/singular-systems.md): concept page.
- [`gko::preconditioner::NullspaceProjection`](../preconditioner/nullspace-projection.md): the preconditioner wrapper that applies the projection.
:::
