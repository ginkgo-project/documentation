# Move data between executors

When you need to copy a Ginkgo object from one executor to another,
typically host ↔ GPU, there are three idioms. They look similar but
they don't all do the same thing.

## The three options

```cpp
// Deep copy returning a new object on `target`, typed like A.
auto B = gko::clone(target, A);

// Method form on a concrete cloneable type: also typed like A.
auto B_typed = A->clone(target);

// Copy in place. B already exists on `target`; overwrite its contents.
B->copy_from(A);
```

| Idiom | What it does | When to use |
|---|---|---|
| `gko::clone(exec, obj)` | Allocates a new object on `exec` and deep-copies contents into it. Works on any [`Cloneable`](../../../api-reference/base/cloneable.md) object and **preserves the static type** of `obj`: a `shared_ptr<Csr<>>` clones to a `unique_ptr<Csr<>>`. | You want a fresh, *typed* copy on the target executor, also when you only hold a `LinOp` pointer. |
| `obj->clone(exec)` | Member of the cloneable concrete types (matrices, vectors, partitions; classes deriving from `EnableCloneable<T>`). It returns a `unique_ptr<T>` of the concrete type. Through a `LinOp*`, use `gko::clone(exec, ptr)` or `gko::as<gko::Cloneable>(ptr)->clone(exec)`, which returns a `unique_ptr<gko::Cloneable>`. | You hold the concrete type and want a copy on another executor. |
| `dst->copy_from(src)` | `Cloneable` member that copies `src`'s contents into the existing `dst`, under the same conversion rule: `src` must be convertible to `dst`'s type. | You want to reuse an existing buffer (e.g. inside an iteration loop) without reallocating. |

The header doxygen for `gko::clone` calls out the type-preservation
point explicitly: "the difference between this function and directly
calling `Cloneable::clone()` is that this one preserves the static type of
the object." That makes `gko::clone` the preferred form whenever you
want to keep using the concrete matrix or vector type without an
intermediate cast.

Cloning covers data objects. To run a solver or preconditioner on another
executor, build its factory with `.on(target_exec)` and `generate` from the
matrix on that executor. A factory from `.on(exec)` is a `unique_ptr`; use
`gko::share` or `std::move` to hand it on.

## Plain `gko::array`

`array` has its own cross-executor constructor, which takes the target
executor and the source array. Copy and move assignment between arrays on
different executors copy the data as well:

```cpp
gko::array<int> a_host{host_exec, {1, 2, 3, 4}};
gko::array<int> a_dev{device_exec, a_host};        // copy across
a_dev = a_host;                                    // assignment also copies across
```

## Skip the copy when possible

- **Same executor instance.** A copy inside one executor stays in that
  executor's memory and does not cross the PCIe boundary.
- **Memory-accessible executors.** `exec->memory_accessible(other)`
  reports whether two executors share the same memory, for example a
  host executor and a device executor that uses unified memory. Data in
  shared memory is directly usable from both executors.

For non-owning views, where you want to *wrap* an existing buffer
without copying at all, see
[Zero-copy from application memory](zero-copy-application-memory.md).

## Common pitfalls

- **`copy_from` requires the destination to already exist.** Build it on
  the target executor first (`Csr::create(target)`, `Dense::create(target, dim)`).
- **`copy_from` between different formats does conversion.** Going from
  COO to CSR via `copy_from` runs the format conversion kernel, not a
  pure memcpy.
- **`apply(b, x)` with operands on another executor.** Ginkgo clones
  them to the operator's executor internally, which is correct but wasteful
  inside loops. Move the operands once, before the loop.

:::{seealso}
- [Switch executor](switch-executor.md): set up the target executor in the first place.
- [Executor model](../../concepts/executor-model.md): the conceptual reference.
:::
