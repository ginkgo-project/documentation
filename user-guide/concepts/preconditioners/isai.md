# ISAI (Incomplete Sparse Approximate Inverse)

`gko::preconditioner::Isai<IsaiType, ValueType, IndexType>` computes a
**sparse approximation** $M_A \approx A^{-1}$ directly, then applies it as
the preconditioner via a single sparse matrix-vector product. The apply
step stays free of triangular solves, which is exactly the appeal on
hardware where triangular substitution serialises poorly.

The trade-off is that the quality of the preconditioner is bounded by the
sparsity pattern it is constrained to. ISAI fixes the pattern of $M_A$ to
match that of $A$ (or one of its triangular factors), then solves a
collection of small dense systems to choose the entries that minimise the
local error.

Four ISAI flavours are selectable through the `IsaiType` template
parameter:

| `IsaiType` | Approximates | Typical pairing |
|---|---|---|
| `isai_type::general` | $M_A \approx A^{-1}$ | Standalone preconditioner for general $A$ where a triangular factorisation is unavailable. |
| `isai_type::lower` | $M_L \approx L^{-1}$ | Replaces the lower-triangular solve in an ILU / IC pipeline. |
| `isai_type::upper` | $M_U \approx U^{-1}$ | Replaces the upper-triangular solve in an ILU pipeline. |
| `isai_type::spd` | $M_C^T M_C \approx C^{-1} C^{-T}$ for the Cholesky factor $C$ of $A$ | Symmetric variant: the *Factorized Sparse Approximate Inverse* (FSPAI). |

## When to use ISAI

- **As a replacement for the triangular solves inside an
  [ILU](ilu.md) / [IC](ic.md) preconditioner** when those solves
  serialise too poorly on the target hardware. Build the factorisation as
  usual, then plug `Isai<lower>` and `Isai<upper>` in as the `l_solver` /
  `u_solver` factories.
- **As a standalone preconditioner** for general $A$ when no factorisation
  is available; ISAI's per-iteration cost is just an SpMV.
- **For SPD systems** (`spd` flavour) where a triangular-solve-free
  preconditioner is preferred for parallelism reasons.

ISAI's preconditioner quality is fundamentally limited by the chosen
sparsity pattern; for matrices where the inverse is dense, ISAI cannot
compete with a full factorisation regardless of how much computation you
throw at the generation step. Profile against [IC](ic.md) / [ILU](ilu.md)
with a Krylov inner triangular solve before adopting ISAI.

## Construction

```cpp
// Standalone general ISAI preconditioner
auto isai = gko::preconditioner::Isai<
                gko::preconditioner::isai_type::general,
                double, gko::int32>::build()
                .on(exec);

auto preconditioner = isai->generate(system_matrix);

// As the triangular solver inside an ILU preconditioner
auto ilu = gko::preconditioner::Ilu<
                gko::preconditioner::LowerIsai<double, gko::int32>,
                gko::preconditioner::UpperIsai<double, gko::int32>>::build()
                .with_factorization(
                    gko::factorization::ParIlu<double, gko::int32>::build()
                        .on(exec))
                .on(exec);
```

`LowerIsai` and `UpperIsai` are typedefs for `Isai<isai_type::lower, …>`
and `Isai<isai_type::upper, …>` respectively, supplied as convenience
shorthands when wiring ISAI into ILU.

## Factory parameters

| Parameter | Type | Purpose |
|---|---|---|
| `skip_sorting` | `bool` | Skip sorting the input matrix when it is known to be sorted by row, then by column; the algorithm requires sorted input. Default `false`. |
| `sparsity_power` | `int` | Use the sparsity pattern of $A^k$ instead of $A^1$ for the approximate inverse. Larger values give denser approximations and better preconditioner quality at the cost of generation time and storage. Default `1`. |
| `excess_limit` | `size_type` | Size limit for the excess system (see below). When set to some $m > 0$, the excess system is solved as soon as its size exceeds $m$, repeatedly until the full excess solution is computed. Default `0`. |
| `excess_solver_factory` | `LinOpFactory` | Solver factory for the excess system. Defaults to a triangular solver for `lower` and `upper` ISAI, and to block-Jacobi preconditioned GMRES for `general` and `spd` ISAI. |
| `excess_solver_reduction` | `remove_complex<ValueType>` | Residual-reduction factor of the default GMRES excess solver used for `general` and `spd` ISAI. Only the default GMRES excess solver (`general` / `spd`) reads it. Default `1e-6`. |

## Implementation notes

Rows with more than 32 nonzero entries produce an *excess system*. Ginkgo
collects these systems and solves them with sparse triangular solves (for
`lower` and `upper` ISAI) or with GMRES (for `general` and `spd` ISAI),
configured through `excess_solver_factory`; `excess_solver_reduction` sets
the residual-reduction target of the default GMRES.
`excess_limit` bounds the size of the collected system. Since a larger
`sparsity_power` widens the rows, expect more excess systems when the
matrix already has many entries per row.

In the general case, $\mathrm{ISAI}(A)^T \ne \mathrm{ISAI}(A^T)$; only
the `spd` flavour is transpose-symmetric. If you need the transposed approximate inverse, compute it from
$A^T$ directly.

:::{seealso}
- [IC family](ic.md) and [ILU family](ilu.md): incomplete factorisations
  whose triangular solves ISAI can replace.
- [Solver taxonomy](../solvers.md): Krylov methods that pair with ISAI.
- API reference: [`gko::preconditioner::Isai`](../../../api-reference/preconditioner/isai.md).

References:
- Anzt, H., Huckle, T. K., Bräckle, J., Dongarra, J.
  *Incomplete Sparse Approximate Inverses for Parallel Preconditioning.*
  Parallel Computing, 71, 1–22, 2018.
  <https://doi.org/10.1016/j.parco.2017.10.003>
:::
