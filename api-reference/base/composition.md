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

The classic use case is factorisation output: `factorization::Ilu`,
`ParIlu` and `ParIlut` return a `Composition{L, U}` of their factors (and
`factorization::Ic`, `ParIc` and `ParIct` a `Composition{L, L^H}`), which
`preconditioner::Ilu` and `preconditioner::Ic` also accept in place of the
system matrix.
Multigrid level generators also expose their coarse correction
$P\,A_c\,R$ as a `Composition` through `get_composition()`.

When an inner operator requires an initial guess, `Composition` supplies
one automatically: the previous operator's output if the current
operator is square, or zero if it is rectangular.

```{doxygenclass} gko::Composition
:members:
```
