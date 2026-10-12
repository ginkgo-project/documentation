# `gko::Cloneable`

`Cloneable` is the interface for objects that can be copied polymorphically.
`clone()` and `clone(exec)` copy the object on its own executor or on `exec`.
`copy_from()` and `move_from()` overwrite an existing object, and
`create_default()` makes an empty object of the same type.

The mixin `EnableCloneable<ConcreteType>` implements this interface from the
concrete type's copy and move assignment, and returns the concrete type from
`clone()` and `create_default()`. Matrix formats, `solver::Nullspace` and
`preconditioner::NullspaceProjection` derive from it. The free function
`gko::clone` accepts smart and plain pointers to cloneable objects.

```cpp
auto A = gko::share(gko::read<gko::matrix::Csr<double, int>>(
    std::ifstream("A.mtx"), exec));
// A copy on the host, whatever executor A lives on:
auto A_host = gko::clone(exec->get_master(), A);
// A copy on A's own executor, typed as the concrete class:
std::unique_ptr<gko::matrix::Csr<double, int>> A_copy = A->clone();
```

```{doxygenclass} gko::Cloneable
:members:
```

```{doxygenclass} gko::EnableCloneable
:members:
```

:::{seealso}
- [Linear operators and composition](../../user-guide/concepts/linop-and-composition.md)
:::
