# GCR (Generalised Conjugate Residual)

`gko::solver::Gcr<ValueType>` is a Krylov method for **general
non-symmetric** systems that, like [GMRES](gmres.md), minimises the
2-norm of the residual over the Krylov subspace. Where GMRES does so through
an Arnoldi basis and a Hessenberg least-squares solve, GCR maintains an
explicit set of search directions $p_0, p_1, \ldots$ chosen so that
$\{A p_k\}$ is mutually orthogonal,

$$
\langle A p_i,\, A p_j \rangle = 0 \quad \text{for } i \ne j.
$$

That orthogonality lets each iteration update the residual with a single
projection along $A p_k$. The trade-off is that the search directions all
have to be remembered to keep them orthogonal, so memory grows linearly
with the iteration count, the same as GMRES.

## When to use GCR

- **Non-symmetric systems where you want GMRES-style residual minimisation
  but prefer working with the search directions explicitly**, e.g. when
  pairing with a Krylov-aware preconditioner, or when an analysis pass
  needs the $A p_k$ vectors directly.
- **A restartable alternative to GMRES** with comparable convergence on
  smooth problems. Set `krylov_dim` (default 100) to bound the working
  memory.

In most situations [GMRES](gmres.md) is faster; its Arnoldi basis and the
small Hessenberg solve are typically cheaper per iteration than GCR's
explicit search-direction orthogonalisation. Reach for GCR when you have a
specific reason to want $A p_k$-orthogonality rather than $V_k$-orthogonality.

For non-symmetric systems where bounded memory matters more than residual
minimisation, [BiCGSTAB](bicgstab.md) is a short-recurrence alternative.

## Construction

```cpp
auto gcr = gko::solver::Gcr<double>::build()
               .with_criteria(
                   gko::stop::Iteration::build()
                       .with_max_iters(1000u)
                       .on(exec),
                   gko::stop::ResidualNorm<double>::build()
                       .with_reduction_factor(1e-10)
                       .on(exec))
               .with_krylov_dim(100u)
               .with_preconditioner(
                   gko::preconditioner::Jacobi<double, gko::int32>::build()
                       .on(exec))
               .on(exec);

auto solver = gcr->generate(system_matrix);
solver->apply(b, x);
```

## Factory parameters

GCR inherits the iterative-solver surface from
`EnablePreconditionedIterativeSolver` and adds one tuning knob,
`krylov_dim`:

| Parameter | Type | Purpose |
|---|---|---|
| `criteria` | vector of `stop::CriterionFactory` | Stopping criteria. Required; see [Stopping criteria](../stopping-criteria.md). |
| `preconditioner` | `LinOpFactory` | Preconditioner factory; rebuilt each `generate()` call. Defaults to identity. |
| `generated_preconditioner` | `LinOp` | An already-generated preconditioner. Bypasses the factory and is used directly. |
| `krylov_dim` | `size_type` | Maximum number of search directions kept before restart. Default `100`. Larger values give better convergence per restart at the cost of linear-in-`krylov_dim` memory and orthogonalisation work. |

## Memory footprint

GCR stores `krylov_dim` search directions $p_k$ and `krylov_dim`
preconditioned-and-applied vectors $A p_k$: two vector arrays of length
$n$ each, both growing with the iteration count up to the restart limit.
After a restart the buffers are reused.

## Implementation notes

The orthogonalisation uses modified Gram-Schmidt. Each iteration costs one
SpMV, one preconditioner apply, $O(k)$ inner products for the
re-orthogonalisation against the previous search directions, and an AXPY
update of the iterate and residual.

:::{seealso}
- [GMRES](gmres.md): the canonical residual-minimising Krylov method;
  usually the better default.
- [BiCGSTAB](bicgstab.md): short-recurrence alternative for
  non-symmetric systems.
- [Solver taxonomy](../solvers.md): where GCR fits among the families.
- [Preconditioner taxonomy](../preconditioners.md): what to pair with GCR.
- API reference: [`gko::solver::Gcr`](../../../api-reference/solver/gcr.md).
:::
