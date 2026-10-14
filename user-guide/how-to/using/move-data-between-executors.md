# Move data between executors

When you need to copy a Ginkgo object from one executor to another,
typically host ↔ GPU, there are three idioms. They look similar but
they don't all do the same thing.

## The three options

```cpp
// Deep copy returning a new object on `target`, typed like A.
auto B = gko::clone(target, A);

// Method form: the same deep copy. On a concrete matrix it returns
// that type; through a LinOp pointer it returns unique_ptr<LinOp>.
auto B_method = A->clone(target);

// Copy in place. B already exists on `target`; overwrite its contents.
B->copy_from(A);
```

| Idiom | What it does | When to use |
|---|---|---|
| `gko::clone(exec, obj)` | Allocates a new object on `exec` and deep-copies contents into it. **Preserves the static type** of `obj`: a `shared_ptr<Csr<>>` clones to a `unique_ptr<Csr<>>`. | You want a fresh, *typed* copy on the target executor and don't want to `as<>`-cast afterwards. |
| `obj->clone(exec)` | Same deep copy. The return type follows the static type of `obj`: a concrete `Csr` clones to `unique_ptr<Csr>`, while a `LinOp*` clones to `unique_ptr<LinOp>`. | You hold the object by its concrete type, or you'll downcast with `gko::as<>` after cloning through a base pointer. |
| `dst->copy_from(src)` | Copies `src`'s contents into the existing `dst`. Requires `src` to be `ConvertibleTo<DstType>`. | You want to reuse an existing buffer (e.g. inside an iteration loop) without reallocating. |

The header doxygen for `gko::clone` states that "the difference between
this function and directly calling `LinOp::clone()` is that this one
preserves the static type of the object." That makes `gko::clone` the
preferred form whenever you hold the object through a smart pointer and
want to keep using its concrete type without an intermediate cast.

## Plain `gko::array`

`array` copies across executors through its constructor; pass the
target executor and the source array:

```cpp
gko::array<int> a_host{host_exec, {1, 2, 3, 4}};
gko::array<int> a_dev{device_exec, a_host};        // copy across
```

Assigning to an existing array (`a_dev = a_host;`) copies the contents into
that array's own executor as well.

## Check whether a copy is needed

Two situations keep a copy cheap:

- **Same executor instance.** A copy within one executor stays in that
  executor's memory, so no transfer crosses the PCIe boundary.
- **Executors that share memory.** `exec->memory_accessible(other)`
  reports whether two executors can access the same memory (for example
  two OpenMP executors, or an OpenMP executor and a DPC++ executor on a
  CPU device). Ginkgo uses it to skip the clone when it needs an object
  on a different executor.

For non-owning views, where you want to *wrap* an existing buffer
without copying at all, see
[Zero-copy from application memory](zero-copy-application-memory.md).

## Common pitfalls

- **`copy_from` requires the destination to already exist.** Build it on
  the target executor first (`Csr::create(target)`, `Dense::create(target, dim)`).
- **`copy_from` between different formats does conversion.** Going from
  COO to CSR via `copy_from` runs the format conversion kernel, not a
  pure memcpy.
- **`apply(b, x)` converts executors for you.** Operands on another
  executor are cloned to the operator's executor internally when the two
  executors do not share memory. The result is correct, but the repeated
  transfers are wasteful inside loops, so keep `b` and `x` on the
  operator's executor.

:::{seealso}
- [Switch executor](switch-executor.md): set up the target executor in the first place.
- [Executor model](../../concepts/executor-model.md): the conceptual reference.
:::
