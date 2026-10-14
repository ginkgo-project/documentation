# IDR(s): Induced Dimension Reduction

`gko::solver::Idr<ValueType>` is a short-recurrence Krylov method for
**general non-symmetric** systems. Unlike [GMRES](gmres.md), whose memory
grows linearly with the iteration count, IDR(s) keeps a fixed shadow
subspace of size $s$ throughout the run, giving GMRES-like convergence
robustness with bounded memory.

The method is built around the *induced dimension reduction* theorem.
Fix a full-rank shadow space
$R = \mathrm{span}(r_1, \ldots, r_s)$ of $s$ random orthonormal vectors.
The theorem guarantees the existence of a sequence of nested subspaces

$$
\mathcal{G}_{j+1} = (I - \omega_j A)\, (\mathcal{G}_j \cap R^{\perp}),
\qquad \mathcal{G}_0 = \mathbb{R}^N,
$$

each strictly contained in its predecessor. IDR(s) forces successive
residuals into these shrinking spaces; in exact arithmetic they become
identically zero after at most $N + N/s$ iterations, substantially fewer
matrix-vector products than the $2N$ BiCG would require.

## When to use IDR(s)

- **Non-symmetric problems where bounded memory matters**: large 3D
  unsteady flow simulations, time-stepping with millions of unknowns,
  contexts where restarted GMRES stalls but explicit memory budgets rule
  out keeping a growing Krylov basis.
- **Hard non-symmetric matrices** where [BiCGSTAB](bicgstab.md) stalls.
  IDR(s) with $s = 4$ or $s = 8$ often converges where BiCGSTAB does
  not, at the cost of more work per iteration.
- **Distributed-memory runs** where the smaller communication footprint
  per iteration (compared with GMRES) trades favourably for the extra
  shadow-space arithmetic.

For typical non-symmetric problems where memory is not a constraint,
[GMRES](gmres.md) usually converges faster per matvec; [BiCGSTAB](bicgstab.md)
is cheaper per iteration and usually a better first try.

## Construction

```cpp
auto idr = gko::solver::Idr<double>::build()
               .with_criteria(
                   gko::stop::Iteration::build()
                       .with_max_iters(1000u)
                       .on(exec),
                   gko::stop::ResidualNorm<double>::build()
                       .with_reduction_factor(1e-10)
                       .on(exec))
               .with_subspace_dim(4u)
               .with_preconditioner(
                   gko::preconditioner::Jacobi<double, gko::int32>::build()
                       .on(exec))
               .on(exec);

auto solver = idr->generate(system_matrix);
solver->apply(b, x);
```

## Factory parameters

In addition to the standard iterative-solver parameters, IDR(s) exposes
four IDR-specific knobs (the last four rows):

| Parameter | Type | Purpose |
|---|---|---|
| `criteria` | vector of `stop::CriterionFactory` | Stopping criteria. Required; see [Stopping criteria](../stopping-criteria.md). |
| `preconditioner` | `LinOpFactory` | Preconditioner factory; rebuilt each `generate()` call. Defaults to identity. |
| `generated_preconditioner` | `LinOp` | An already-generated preconditioner. Bypasses the factory. |
| `subspace_dim` | `size_type` | Shadow-space dimension $s$. Default `2`. Larger $s$ improves convergence at the cost of more inner products and storage per iteration. Typical choices: `2`, `4`, `8`. |
| `kappa` | `remove_complex<ValueType>` | Perpendicularity threshold: $A v_n$ and $v_n$ are treated as too close to perpendicular when $\lvert (Av_n)^H v_n / (\lVert Av_n \rVert \lVert v_n \rVert) \rvert < \kappa$. Default `0.7`. |
| `deterministic` | `bool` | If `true`, the vectors spanning the shadow space are chosen deterministically instead of randomly. Default `false`. They are then generated serially on the CPU, which can be very slow, so this mode is mainly for testing. |
| `complex_subspace` | `bool` | If `true`, IDR uses a complex shadow space also for real problems, which can give faster convergence by accounting for complex eigenvectors. Default `false`. |

## Memory footprint

IDR(s) stores $O(s)$ vectors at fixed cost throughout the run: the
shadow space $R$, $s$ search-direction vectors, $s$ "subspace-update"
vectors, plus a few iteration-state vectors. Total: roughly $4s + 4$
vectors of length $n$. The memory cost is independent of iteration count.

## Implementation notes

The implementation follows Van Gijzen and Sonneveld's *Algorithm 913*,
which packs the shadow-space orthogonalisation into BLAS-3-style block
operations against a dense $s \times s$ inner-product matrix. On GPUs the
block form gives noticeably better arithmetic intensity than per-vector
inner products.

:::{seealso}
- [GMRES](gmres.md): residual-minimising alternative with growing memory.
- [BiCGSTAB](bicgstab.md): short-recurrence non-symmetric alternative;
  usually cheaper per iteration but less robust on hard problems.
- [Solver taxonomy](../solvers.md): where IDR(s) fits among the families.
- [Preconditioner taxonomy](../preconditioners.md): what to pair with IDR(s).
- API reference: [`gko::solver::Idr`](../../../api-reference/solver/idr.md).

Reference:
Van Gijzen, M. B., Sonneveld, P. *Algorithm 913: An Elegant IDR(s) Variant
that Efficiently Exploits Biorthogonality Properties.* ACM Transactions on
Mathematical Software, 38 (1), Article 5, 2011.
<https://doi.org/10.1145/2049662.2049667>
:::
