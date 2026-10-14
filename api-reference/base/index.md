# Core types

The abstract interfaces every Ginkgo class lives behind. `LinOp` is the
linear-operator interface: matrices, solvers, preconditioners, and any
composed operator inherits from it. `LinOpFactory` is the matching
factory interface that generates a `LinOp` from a system matrix. The
remaining types in this section are mixins and trait classes that
concrete operators implement.

## The operator interface

- [`LinOp`](lin-op.md): abstract base for every linear operator in
  Ginkgo. Defines `apply`, size, executor, and the logging hooks that
  flow through every operation.
- [`LinOpFactory`](lin-op-factory.md): abstract base for factories
  that produce a `LinOp` (typically a solver or preconditioner) when
  given a system matrix.

## Operator algebra

- [`Combination`](combination.md): a linear combination of `LinOp`s,
  $c_1\, A_1 + c_2\, A_2 + \cdots + c_k\, A_k$, as a single `LinOp`.
- [`Composition`](composition.md): function composition of `LinOp`s,
  $A_1 \cdot A_2 \cdot \ldots \cdot A_n$, evaluated right-to-left.

## Memory containers

- [`array`](array.md): owning, executor-aware contiguous buffer; the
  storage unit that backs every Ginkgo data structure.
- [`matrix_data`](matrix-data.md): host-side `(row, col, value)`
  triple-list used as the executor-independent assembly format for
  every sparse matrix type.

## Mixins and traits

- [`PolymorphicObject`](polymorphic-object.md): the base of every Ginkgo
  object; it holds the executor and provides `validate_data()`.
- [`Cloneable` / `EnableCloneable`](cloneable.md): interface for objects
  that can be cloned, copied and moved polymorphically (`clone()`,
  `clone(exec)`, `copy_from`, `move_from`, `create_default`), and the mixin
  that implements it for a concrete type. Matrices implement it;
  `gko::clone(obj)` / `gko::clone(exec, obj)` works on any cloneable object.
- [`Transposable`](transposable.md): trait interface for operators that
  can produce their transpose and conjugate transpose.

## Errors

- [Exceptions](exceptions.md): `gko::Error` and the exception types Ginkgo
  throws.

```{toctree}
:hidden:
:maxdepth: 1

lin-op
lin-op-factory
combination
composition
array
matrix-data
transposable
polymorphic-object
cloneable
exceptions
```
