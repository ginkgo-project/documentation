# Move data between executors

When you need to copy a Ginkgo object from one executor to another —
typically host ↔ GPU — there are three idioms. They look similar but
they don't all do the same thing.

## The three options

```cpp
// Deep copy returning a new object on `target`, typed like A.
auto B = gko::clone(target, A);

// Method form: same deep copy, but the return type is the base
// polymorphic type (e.g. unique_ptr<LinOp>, not unique_ptr<Csr<...>>).
auto B_base = A->clone(target);

// Copy in place. B already exists on `target`; overwrite its contents.
B->copy_from(A);
```

| Idiom | What it does | When to use |
|---|---|---|
| `gko::clone(exec, obj)` | Allocates a new object on `exec` and deep-copies contents into it. **Preserves the static type** of `obj` — a `shared_ptr<Csr<>>` clones to a `unique_ptr<Csr<>>`. | You want a fresh, *typed* copy on the target executor and don't want to `as<>`-cast afterwards. |
| `obj->clone(exec)` | Same deep copy, but returns the base type the `clone` method declares (`unique_ptr<LinOp>` / `unique_ptr<PolymorphicObject>`). | You're working through a base pointer anyway, or you'll downcast with `gko::as<>` next. |
| `dst->copy_from(src)` | Copies `src`'s contents into the existing `dst`. Requires `src` to be `ConvertibleTo<DstType>`. | You want to reuse an existing buffer (e.g. inside an iteration loop) without reallocating. |

The header doxygen for `gko::clone` calls out the type-preservation
point explicitly: "the difference between this function and directly
calling `LinOp::clone()` is that this one preserves the static type of
the object." That makes `gko::clone` the preferred form whenever you
want to keep using the concrete matrix or vector type without an
intermediate cast.

## Plain `gko::array`

`array` has its own cross-executor constructor — pass the target
executor and the source array:

```cpp
gko::array<int> a_host{host_exec, {1, 2, 3, 4}};
gko::array<int> a_dev{device_exec, a_host};        // copy across
```

There is no `array::copy_to_executor()` — use the constructor above or
`gko::clone(exec, a)` for the generic form.

## Skip the copy when possible

Two cases where `copy_from` is a no-op:

- **Same executor instance.** Copying inside the same executor still
  goes through device memcpy semantics, but no transfer crosses the
  PCIe boundary.
- **Memory-accessible executors.** When the source and destination both
  use unified memory or pinned host memory, the runtime can resolve the
  copy without touching the data. See `Executor::memory_accessible` for
  the rule.

For non-owning views — where you want to *wrap* an existing buffer
without copying at all — see
[Zero-copy from application memory](zero-copy-application-memory.md).

## Common pitfalls

- **`copy_from` requires the destination to already exist.** Build it on
  the target executor first (`Csr::create(target)`, `Dense::create(target, dim)`).
- **`copy_from` between different formats does conversion.** Going from
  COO to CSR via `copy_from` runs the format conversion kernel, not a
  pure memcpy.
- **`apply(b, x)` does *not* convert executors silently.** Operand
  executors must match the operator's. Mismatches trigger an internal
  clone — correct, but wasteful inside loops.

:::{seealso}
- [Switch executor](switch-executor.md) — set up the target executor in the first place.
- [Executor model](../../concepts/executor-model.md) — the conceptual reference.
:::
