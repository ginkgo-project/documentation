# `gko::Combination`

Linear combination of `LinOp`s. Given $k$ operators
$\mathit{op}_1, \ldots, \mathit{op}_k$ and $k$ scalar coefficients
$c_1, \ldots, c_k$ (each a $1 \times 1$ `Dense`), `Combination` represents

$$
\mathit{Combination}(x) = c_1\, \mathit{op}_1\, x + c_2\, \mathit{op}_2\, x
                          + \cdots + c_k\, \mathit{op}_k\, x,
$$

and is itself a `LinOp`. `apply` runs each constituent and accumulates the
scaled results into the output. The constructor moves all operators onto
the executor of the first one if their executors differ, so the combined
operator runs on a single executor.

Common use cases: assembling $A - \lambda I$ as `Combination{1.0, A,
-lambda, I}` for shifted-system solves, building residual operators of
the form `b - A x` as a single `LinOp` that a logger or downstream tool
can consume, and stacking multiple corrections in iterative refinement
pipelines.

```{doxygenclass} gko::Combination
:members:
```
