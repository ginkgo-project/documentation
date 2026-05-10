# Add a new iterative solver

:::{note}
Stub — content forthcoming.
:::

The CRTP skeleton for an iterative solver: `EnableLinOp`,
`EnableSolverBase`, `workspace_traits` specialisation, the
`apply_dense_impl` flow with `GKO_SOLVER_VECTOR` / `GKO_SOLVER_SCALAR`
boilerplate macros, precision dispatch, and the distributed-ready
`precision_dispatch_real_complex_distributed` overload.
