# `gko::EnableLinOp`

CRTP mixin that supplies the boilerplate every concrete `LinOp` needs:
the `apply` dispatcher (which calls into the derived class's
`apply_impl`), the default `clone` / `copy_from` plumbing, the
polymorphic-object machinery, and the logging hooks. Concrete
operators (every solver, preconditioner, and matrix in Ginkgo)
derive from `EnableLinOp<MyOperator>` rather than from `LinOp`
directly: the mixin's CRTP cast lets these defaults work on the
derived type without virtual-dispatch overhead.

```{doxygenclass} gko::EnableLinOp
:members:
```
