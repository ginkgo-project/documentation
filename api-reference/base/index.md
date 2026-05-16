# Core types

The abstract interfaces every Ginkgo class lives behind. `LinOp` is the
linear-operator interface — matrices, solvers, preconditioners, and any
composed operator inherits from it. `LinOpFactory` is the matching
factory interface that generates a `LinOp` from a system matrix. The
remaining types in this section are mixins and trait classes that
concrete operators implement.

## The operator interface

- [`LinOp`](lin-op.md) — abstract base for every linear operator in
  Ginkgo. Defines `apply`, copy/clone, size, executor, and the logging
  hooks that flow through every operation.
- [`LinOpFactory`](lin-op-factory.md) — abstract base for factories
  that produce a `LinOp` (typically a solver or preconditioner) when
  given a system matrix.

## Operator algebra

- [`Combination`](combination.md) — a linear combination of `LinOp`s,
  $c_1\, A_1 + c_2\, A_2 + \cdots + c_k\, A_k$, as a single `LinOp`.
- [`Composition`](composition.md) — function composition of `LinOp`s,
  $A_1 \cdot A_2 \cdot \ldots \cdot A_n$, evaluated right-to-left.

## Memory containers

- [`array`](array.md) — owning, executor-aware contiguous buffer; the
  storage unit that backs every Ginkgo data structure.
- [`matrix_data`](matrix-data.md) — host-side `(row, col, value)`
  triple-list used as the executor-independent assembly format for
  every sparse matrix type.

## Mixins and traits

- [`EnableLinOp`](enable-linop.md) — CRTP mixin that supplies the
  boilerplate `LinOp` overrides — `apply_impl` dispatch, default
  copy/clone, polymorphic-object plumbing — so concrete operators only
  need to implement the algorithm.
- [`Transposable`](transposable.md) — trait interface for operators that
  can produce their transpose and conjugate transpose.

```{toctree}
:hidden:
:maxdepth: 1

lin-op
lin-op-factory
combination
composition
array
matrix-data
enable-linop
transposable
```
