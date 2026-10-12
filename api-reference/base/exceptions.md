# Exceptions

Every Ginkgo exception derives from `gko::Error`, which carries the file, line
and message and is itself a `std::runtime_error`. Catching `gko::Error` handles the
exception types listed below.

| Exception | Thrown when |
|---|---|
| `NotSupported` | it is not possible to perform the requested operation on the given object type |
| `NotCompiled` | an operation is called that is part of a module that was not compiled on the system |
| `NotImplemented` | an operation has not yet been implemented (but will be implemented in the future) |
| `DimensionMismatch` | an operation is applied to `LinOp`s of incompatible size |
| `BadDimension` | an operation is applied to a `LinOp` with bad dimensions |
| `ValueMismatch` | two values are not equal |
| `AllocationError` | a memory allocation fails |
| `OutOfBoundsError` | a memory access is detected to be out of bounds |
| `InvalidStateError` | an object is in an invalid state |
| `InvalidData` | an object contains invalid data, described by the invariant it violated; raised by [`validate_data()`](polymorphic-object.md) |

```{doxygenclass} gko::Error
:members:
```

```{doxygenclass} gko::NotSupported
```

```{doxygenclass} gko::NotCompiled
```

```{doxygenclass} gko::NotImplemented
```

```{doxygenclass} gko::DimensionMismatch
```

```{doxygenclass} gko::BadDimension
```

```{doxygenclass} gko::ValueMismatch
```

```{doxygenclass} gko::AllocationError
```

```{doxygenclass} gko::OutOfBoundsError
```

```{doxygenclass} gko::InvalidStateError
```

```{doxygenclass} gko::InvalidData
```

:::{seealso}
- [`gko::PolymorphicObject`](polymorphic-object.md)
:::
