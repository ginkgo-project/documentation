# Integrate Ginkgo with PETSc

:::{note}
Stub — content forthcoming.
:::

Use a Ginkgo solver as a PETSc `PCSHELL` preconditioner: share the matrix
data via zero-copy, wrap PETSc `Vec` objects in Ginkgo `Dense`, and route
the apply through a Ginkgo solver factory. Includes notes on lifetime,
executor placement, and pitfalls when PETSc is built with its own GPU
backend.
