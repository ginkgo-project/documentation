# Singular systems and nullspaces

A singular system $A x = b$ has a nontrivial nullspace $N(A)$, so the solution is determined only up to an element of $N(A)$. Ginkgo's `gko::solver::Nullspace<ValueType>` describes that nullspace and acts as the orthogonal projector onto its complement. Passing it to a Krylov solver through the `nullspace` factory parameter keeps the iterates free of nullspace components, so the solver converges to the minimum-norm solution.

## When a system is singular

A singular system has a solution only if the right-hand side is consistent, that is, if $b$ lies in $\mathrm{range}(A)$. The solution is then unique up to $N(A)$, and the minimum-norm solution is the one orthogonal to $N(A)$.

Two classic sources of singular systems:

- **Pure Neumann Poisson problems.** Without a Dirichlet condition the potential is defined up to an additive constant, so $N(A)$ is spanned by the constant vector.
- **Linear elasticity without supports.** A free body can translate and rotate without deforming, so $N(A)$ is spanned by the rigid-body modes (three translations and three rotations in 3D).

A Krylov method never reduces error components in $N(A)$, because $A$ annihilates them. Rounding errors, a right-hand side that is only approximately consistent, and a preconditioner that does not respect the nullspace all feed such components into the iterates, and nothing damps them again. They can therefore grow over the iterations. Removing them explicitly keeps the iterates clean.

## Describing the nullspace

`gko::solver::Nullspace<ValueType>` is created directly with one of two static functions:

`create_from_constant(exec)` describes a constant-only nullspace, which is what a pure Neumann problem needs. The constant vector is never stored; projecting subtracts the mean. One such object works for any matrix size, any value type and any partition of a distributed vector, so you can share it between all levels of a multigrid hierarchy.

`create(exec, basis, contains_constant)` takes explicit basis vectors, each entry an $n \times k_i$ `matrix::Dense` (or a distributed vector). The basis is copied and orthonormalized once, and numerically dependent columns are dropped, so the basis does not have to be orthonormal or independent. Set `contains_constant` to `true` when the constant vector is part of the nullspace as well. `get_dimension()` reports how many orthonormal vectors remain, plus one for the constant.

```cpp
// Explicit basis vectors (here the columns of `modes`), plus the constant:
auto nullspace = gko::share(gko::solver::Nullspace<double>::create(
    exec, {modes}, /* contains_constant */ true));
std::cout << "nullspace dimension: " << nullspace->get_dimension() << '\n';
```

The `Nullspace` object does not change after creation, and solvers work on their own copy that shares the basis. One instance can therefore be passed to any number of factories, but a single instance must not be applied from several threads at the same time.

## Solvers that remove the nullspace

`Cg`, `Fcg`, `Minres`, `Gmres` and `Bicgstab` accept two factory parameters:

- `nullspace` is the right nullspace $N(A)$. The solver removes its component from the initial guess and wraps its preconditioner in a [`NullspaceProjection`](#projecting-inside-a-preconditioner), so the preconditioned vector stays orthogonal to $N(A)$. For a consistent right-hand side the solver converges to the minimum-norm solution.
- `left_nullspace` is the left nullspace $N(A^H)$. The solver projects $b$ onto $\mathrm{range}(A)$, which makes an inconsistent system consistent, so it is solved in the least-squares sense. Stopping criteria and loggers then see the projected right-hand side.

For the Hermitian solvers `Cg`, `Fcg` and `Minres`, the left and right nullspace coincide, so setting either one sets both. `Gmres` and `Bicgstab` treat them independently: set `left_nullspace` together with `nullspace` whenever $b$ may have a component outside $\mathrm{range}(A)$, otherwise these two solvers can stagnate or diverge. `Minres` applies the projection on both sides of the preconditioner, which keeps the preconditioner symmetric.

```cpp
// Pure Neumann boundary conditions: the constant vector spans the nullspace.
auto nullspace = gko::share(
    gko::solver::Nullspace<double>::create_from_constant(exec));
auto solver =
    gko::solver::Cg<double>::build()
        .with_nullspace(nullspace)
        .with_preconditioner(gko::preconditioner::Jacobi<double, int>::build())
        .with_criteria(gko::stop::max_iters(1000u),
                       gko::stop::relative_residual_norm(1e-8))
        .on(exec)
        ->generate(A);
solver->apply(b, x);  // x has no component in the nullspace
```

`Bicg`, `Cgs`, `Gcr`, `Idr`, `PipeCg`, `CbGmres` and `Chebyshev` have the two parameters as well but throw `NotSupported` at `generate()` when a nullspace is set. Nullspace removal works on every executor and with distributed matrices; an explicit basis then has to consist of distributed vectors on the partition of the system matrix. Inside a multigrid hierarchy, give the coarsest-level solver the same constant nullspace as the outer solver.

## Projecting inside a preconditioner

Solvers create a `gko::preconditioner::NullspaceProjection<ValueType>` for you, but you can also build one yourself and use it as the preconditioner of any solver. It computes $z = P M Q r$: it applies the wrapped preconditioner $M$, removes the right nullspace ($P$) from the result and optionally removes the left nullspace ($Q$) from the input. When you use it by hand, project $b$ and the initial guess yourself with `Nullspace::project`.

```cpp
auto precond =
    gko::preconditioner::NullspaceProjection<double>::build()
        .with_preconditioner(gko::preconditioner::Jacobi<double, int>::build())
        .with_nullspace(nullspace)
        .with_left_nullspace(nullspace)
        .on(exec)
        ->generate(A);
```

For a symmetric problem, pass the same object as `nullspace` and `left_nullspace`.

## Configuration files

In a JSON configuration, `"nullspace"` and `"left_nullspace"` are strings that name a `Nullspace` object stored in the registry. The `"preconditioner::NullspaceProjection"` type accepts the keys `preconditioner` (an inline configuration or a registry name), `generated_preconditioner`, `nullspace` and `left_nullspace`.

Store the nullspace in a registry under the name used in the JSON, then parse the property tree `config` (read from the JSON below) with that registry:

```cpp
gko::config::registry reg;
auto ns = gko::share(
    gko::solver::Nullspace<double>::create_from_constant(exec));
reg.emplace("ns", ns);
// `config`: the pnode parsed from the JSON below;
// `type_desc`: the gko::config::type_descriptor in use
auto factory = gko::config::parse(config, reg, type_desc).on(exec);
auto solver = factory->generate(A);
solver->apply(b, x);
```

```json
{
  "type": "solver::Cg",
  "nullspace": "ns",
  "criteria": {"iteration": 1000, "relative_residual_norm": 1e-10}
}
```

See [Configuration files](../config-files.md) for the general mechanics of the registry.

:::{seealso}
- [Solver taxonomy](../solvers.md): the Krylov solvers that accept a nullspace.
- [CG](cg.md), [GMRES](gmres.md): factory-parameter tables with `nullspace` and `left_nullspace`.
- API reference: [`gko::solver::Nullspace`](../../../api-reference/solver/nullspace.md), [`gko::preconditioner::NullspaceProjection`](../../../api-reference/preconditioner/nullspace-projection.md)
:::
