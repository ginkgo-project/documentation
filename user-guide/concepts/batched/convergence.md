# Batched convergence logger

`gko::batch::log::BatchConvergence<ValueType>` is the per-item observability hook for batched
solvers. It captures the **iteration count** and **final residual norm** for every item in the
batch and exposes them as host-readable arrays after the solve. Without this logger attached,
per-item convergence information is computed during the solve but discarded — the only
observable signal is the global success of `solver->apply`.

## What it captures

After a batched solver completes, the logger holds two arrays of length $N$ (one entry per
batch item):

| Array              | Type                          | Accessor                  |
| ------------------ | ----------------------------- | ------------------------- |
| Iteration counts   | `gko::array<int>`             | `get_num_iterations()`    |
| Final residuals    | `gko::array<real_type>`       | `get_residual_norm()`     |

`real_type` is `gko::remove_complex<ValueType>` — i.e. `float`/`double` for real solvers and
the real part type for complex ones.

The "final residual" is the **implicit** residual computed inside the solver process — the same
quantity the stopping criterion compares against the tolerance. Depending on the algorithm
(CG, BiCGSTAB) the implicit residual can drift from the true $\lVert b - A x \rVert$, so for
strict accuracy claims compute the true residual yourself with an extra SpMV after the solve.

## Usage

```cpp
namespace b = gko::batch;

// Construct the solver and an empty logger.
auto solver = b::solver::Cg<double>::build()
                  .with_max_iterations(500)
                  .with_tolerance(1e-8)
                  .on(exec)
                  ->generate(batched_matrix);

auto logger = b::log::BatchConvergence<double>::create();
solver->add_logger(logger);

solver->apply(batched_rhs, batched_x);

// Pull per-item results back to the host.
const auto& iters = logger->get_num_iterations();        // gko::array<int>
const auto& norms = logger->get_residual_norm();         // gko::array<double>

// Both arrays live on the executor the logger was created on.
auto host_iters = gko::array<int>{exec->get_master(), iters};
for (gko::size_type k = 0; k < host_iters.get_size(); ++k) {
    std::cout << "item " << k
              << ": iters=" << host_iters.get_const_data()[k] << "\n";
}
```

A logger instance is a one-shot record of a single solve: each call to `solver->apply` resets
the internal arrays. If you want to compare runs, copy the arrays out before the next apply.

## Where the logger fires

`BatchConvergence` listens for the `batch_solver_completed` event — that is the only event in
its default mask. The mask is configurable through the `create(mask)` overload, but no
additional batch-side events are wired up today, so the default is the only useful setting.

The event itself is emitted from inside the fused solver kernel's host-side completion path.
The arrays are populated **after** every item has converged or hit `max_iterations`; there is
no per-iteration logging on the batched path because that would force a host round-trip every
iteration (and defeat the kernel-fusion design).

## Pre-allocation

The logger's arrays are allocated lazily on the first `on_batch_solver_completed` call; the
buffer is sized to match the batch and reused on subsequent solves of the same size. Resizing
across batch sizes is automatic but allocates fresh memory.

:::{seealso}
- [Batched CG](cg.md) and [Batched BiCGSTAB](bicgstab.md) — the solvers this logger observes.
- [Logging and observability](../logging.md) — the broader logger framework on the
  single-system side.
- API reference: [`gko::batch::log::BatchConvergence`](../../../api-reference/batch/log-batch-convergence.md)
:::
