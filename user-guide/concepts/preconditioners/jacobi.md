# Jacobi (scalar and block)

`gko::preconditioner::Jacobi<ValueType, IndexType>` is the block-Jacobi
preconditioner: it inverts the diagonal blocks of the system matrix and
applies that block-diagonal inverse as the approximate inverse of the full
operator. At block size 1 it degenerates to *pointwise* (scalar) Jacobi —
element-wise diagonal scaling. At larger block sizes (up to 32 rows /
columns per block) it captures local couplings that scalar Jacobi misses.

The defining property is locality: each block's inverse is computed and
applied independently, so the preconditioner is embarrassingly parallel.
Block-Jacobi is the standard cheap default for iterative solvers — and the
canonical local solver inside the [Schwarz](../distributed/schwarz.md)
preconditioner for distributed-memory runs.

## When to use Jacobi

- **As a cheap starting preconditioner** for any iterative solver — the
  identity-preconditioner baseline rarely beats it, and the overhead is
  minimal.
- **At block size 1** for diagonally-dominant systems where local couplings
  are weak and pointwise scaling is enough.
- **At larger block sizes** when the matrix has a natural block structure —
  finite-element matrices with multiple DOFs per node, vector-PDE
  discretisations, or any matrix where local couplings within a block are
  much stronger than couplings between blocks.
- **As the local solver inside Schwarz** for distributed-memory problems —
  see [Schwarz](../distributed/schwarz.md).

For matrices where the diagonal blocks are not a good approximation of the
full inverse — e.g., problems with long-range couplings or large condition
number — Jacobi's per-iteration cost is so low that it often still helps,
but more aggressive preconditioners (IC, ILU, AMG) will reduce the
iteration count much further.

## Construction

```cpp
auto jacobi = gko::preconditioner::Jacobi<double, gko::int32>::build()
                  .with_max_block_size(8u)
                  .on(exec);

auto preconditioner = jacobi->generate(system_matrix);

// or pass directly to a solver
auto cg = gko::solver::Cg<double>::build()
              .with_preconditioner(
                  gko::preconditioner::Jacobi<double, gko::int32>::build()
                      .with_max_block_size(8u)
                      .on(exec))
              .with_criteria(...)
              .on(exec);
```

If the block-diagonal structure can be determined from the application
(e.g., DOFs per node in a finite-element problem), pass the explicit block
starting rows via `with_block_pointers`. Otherwise the implementation runs
a supervariable-agglomeration heuristic on the natural blocks it detects in
the matrix.

## Factory parameters

| Parameter | Type | Purpose |
|---|---|---|
| `max_block_size` | `uint32` | Upper bound on the per-block dimension (1 to 32 inclusive). Default `32`. Setting `1` activates pointwise-Jacobi specialised kernels for generation (diagonal inversion) and application (scaling). |
| `block_pointers` | `array<index_type>` | Starting row of each diagonal block. Omit to auto-detect via the supervariable-agglomeration heuristic; the heuristic is conservative — explicit pointers from domain knowledge are usually faster and more accurate. |
| `storage_optimization` | `storage_optimization_type` | Adaptive-precision policy: each block can be stored in a lower precision than `ValueType` to reduce bandwidth. Use the constructor that takes a per-block-precision array to enable, or the global form to apply a single reduction policy to every block. |
| `accuracy` | `remove_complex<ValueType>` | Threshold used when `storage_optimization` is adaptive: a block is stored in a lower precision if the resulting rounding error is below this fraction of the block's condition. Default `1e-1`. |
| `aggregate_l1` | `bool` | Generate the preconditioner on $A + \mathrm{diag}(\sum_{k \in \text{off-block}} |A_{ik}|)$ instead of $A$. Useful as a smoother in multigrid; see the upstream docstring for the row-dominance condition. Default `false`. |

The block-detection heuristic and the available adaptive-precision options
are documented in detail on the class docstring; see the API reference
link below for the full surface.

## Implementation notes

Each diagonal block is inverted explicitly by Gauss-Jordan elimination
with column pivoting and stored in a custom interleaved layout that
preserves coalesced GPU access patterns. The pointwise `max_block_size = 1`
specialisation skips the block format entirely and stores the inverse
diagonal in a single array — significantly faster for very large problems
where the regularity of the access matters more than the block coupling.

The adaptive-precision variant analyses each block's condition number at
generation time and chooses a per-block storage precision (`double` →
`float` → `half`, …) that keeps the block-apply rounding error below the
`accuracy` threshold. This reduces the memory traffic during `apply` —
the biggest cost on GPUs — without changing the working precision of the
iterative solver around it.

:::{seealso}
- [Solvers — taxonomy](../solvers.md) — picking the right solver to pair with Jacobi.
- [Preconditioners — taxonomy](../preconditioners.md) — alternatives when Jacobi is too weak.
- [Schwarz](../distributed/schwarz.md) — uses block-Jacobi as its per-rank local solver.
- API reference: [`gko::preconditioner::Jacobi`](../../../api-reference/preconditioner/jacobi.md).

References:
- Anzt, H., Dongarra, J., Flegar, G., Higham, N. J., Quintana-Ortí, E. S.
  *Adaptive Precision in Block-Jacobi Preconditioning for Iterative
  Sparse Linear System Solvers.*
  ACM Transactions on Mathematical Software, 47 (2), Article 14, 2021.
  <https://doi.org/10.1145/3441850>
:::
