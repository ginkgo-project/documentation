# `gko::Composition`

Function composition of `LinOp`s. Given $n$ operators
$\mathit{op}_1, \ldots, \mathit{op}_n$, `Composition` represents

$$
\mathit{Composition}(x)
  = \mathit{op}_1\, \bigl( \mathit{op}_2\, ( \cdots ( \mathit{op}_n\, x ) ) \bigr),
$$

and is itself a `LinOp`. `apply` runs the operators right-to-left,
feeding each one's output as the input to the next. The constructor
moves all operators onto the executor of the first one if their
executors differ.

The classic use case is assembling triangular-solve chains for
factorisation-based preconditioners: an `Ilu` preconditioner is a
`Composition{U_solver, L_solver}` that, applied to $b$, computes $L^{-1}
b$ first and then $U^{-1}(L^{-1} b)$. The same pattern shows up in
`Ic` (lower-then-conjugate-transpose), in multigrid V-cycles, and
anywhere else where the inverse of a product needs to be expressed as
the product of inverses in reverse order.

When an inner operator requires an initial guess, `Composition` supplies
one automatically: the previous operator's output if the current
operator is square, or zero if it is rectangular.

```{doxygenclass} gko::Composition
:members:
```
