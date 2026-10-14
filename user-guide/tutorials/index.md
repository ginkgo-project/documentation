# Tutorials

Long-form, narrative walkthroughs designed to teach Ginkgo by building
something concrete. Each tutorial is a single executable with copy-paste
code, a CMake recipe, and expected output you can verify against your own
run.

## Available tutorials

- [**Poisson on CPU**](poisson-cpu.md): Build a CG + block-Jacobi solver
  for a 1D Poisson problem on the CPU. Walks through CSR assembly, the RHS
  with boundary conditions, the solver factory, and a mesh-refinement
  study that exhibits the textbook $\mathcal{O}(h^2)$ convergence rate.
- [**Poisson on GPU**](poisson-gpu.md): The same problem, ported to the
  GPU with a three-line diff. Covers the host/device executor pattern,
  cross-executor data movement (`gko::clone`), and when GPU pays off vs.
  when CPU is faster.
- [**Heat equation on a distributed grid**](heat-equation-distributed.md):
  Port the upstream 2D heat-conduction example to Ginkgo's distributed
  types. Implicit-Euler time stepping, partition-by-stripes, Schwarz
  preconditioner, and a per-rank-stripes output pattern that a short Python
  script stitches into an animated GIF.

```{toctree}
:maxdepth: 1
:hidden:

poisson-cpu
poisson-gpu
heat-equation-distributed
```
