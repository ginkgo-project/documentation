# Examples

Annotated walkthroughs of every example shipped under `examples/` in the Ginkgo source tree. Each page interleaves the source with explanatory commentary, lists the prerequisite examples it builds on, and reproduces the expected output. The full program is included at the bottom so you can copy it verbatim.

## Navigation graph

Edges flow from each example to the ones that build on it. Follow the arrows to go from the simplest setups toward more advanced material. Click any node to jump to its page.

```{mermaid}
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '20px', 'fontFamily': 'system-ui'}, 'flowchart': {'nodeSpacing': 40, 'rankSpacing': 60, 'useMaxWidth': false, 'htmlLabels': true, 'padding': 16}}}%%
graph TB
  adaptiveprecision_blockjacobi[adaptiveprecision-blockjacobi]
  batched_solver[batched-solver]
  cb_gmres[cb-gmres]
  custom_logger[custom-logger]
  custom_matrix_format[custom-matrix-format]
  custom_stopping_criterion[custom-stopping-criterion]
  distributed_multigrid_preconditioned_solver[distributed-multigrid-preconditioned-solver]
  distributed_multigrid_reuse[distributed-multigrid-reuse]
  distributed_solver[distributed-solver]
  external_lib_interfacing[external-lib-interfacing]
  file_config_solver[file-config-solver]
  ginkgo_overhead[ginkgo-overhead]
  ginkgo_ranges[ginkgo-ranges]
  heat_equation[heat-equation]
  ilu_preconditioned_solver[ilu-preconditioned-solver]
  inverse_iteration[inverse-iteration]
  ir_ilu_preconditioned_solver[ir-ilu-preconditioned-solver]
  iterative_refinement[iterative-refinement]
  kokkos_assembly[kokkos-assembly]
  minimal_cuda_solver[minimal-cuda-solver]
  mixed_multigrid_preconditioned_solver[mixed-multigrid-preconditioned-solver]
  mixed_multigrid_solver[mixed-multigrid-solver]
  mixed_precision_ir[mixed-precision-ir]
  mixed_spmv[mixed-spmv]
  multigrid_preconditioned_solver[multigrid-preconditioned-solver]
  multigrid_preconditioned_solver_customized[multigrid-preconditioned-solver-customized]
  nine_pt_stencil_solver[nine-pt-stencil-solver]
  papi_logging[papi-logging]
  par_ilu_convergence[par-ilu-convergence]
  performance_debugging[performance-debugging]
  poisson_solver[poisson-solver]
  preconditioned_solver[preconditioned-solver]
  preconditioner_export[preconditioner-export]
  reordered_preconditioned_solver[reordered-preconditioned-solver]
  schroedinger_splitting[schroedinger-splitting]
  simple_solver[simple-solver]
  simple_solver_logging[simple-solver-logging]
  three_pt_stencil_solver[three-pt-stencil-solver]
  preconditioned_solver --> adaptiveprecision_blockjacobi
  simple_solver --> custom_logger
  simple_solver_logging --> custom_logger
  minimal_cuda_solver --> custom_logger
  simple_solver --> custom_matrix_format
  poisson_solver --> custom_matrix_format
  three_pt_stencil_solver --> custom_matrix_format
  simple_solver --> custom_stopping_criterion
  minimal_cuda_solver --> custom_stopping_criterion
  distributed_solver --> distributed_multigrid_preconditioned_solver
  distributed_multigrid_preconditioned_solver --> distributed_multigrid_reuse
  simple_solver --> distributed_solver
  three_pt_stencil_solver --> distributed_solver
  simple_solver --> file_config_solver
  simple_solver --> heat_equation
  three_pt_stencil_solver --> heat_equation
  simple_solver --> ilu_preconditioned_solver
  simple_solver --> inverse_iteration
  ilu_preconditioned_solver --> ir_ilu_preconditioned_solver
  iterative_refinement --> ir_ilu_preconditioned_solver
  simple_solver --> iterative_refinement
  simple_solver --> kokkos_assembly
  poisson_solver --> kokkos_assembly
  three_pt_stencil_solver --> kokkos_assembly
  simple_solver --> minimal_cuda_solver
  multigrid_preconditioned_solver --> mixed_multigrid_preconditioned_solver
  mixed_multigrid_solver --> mixed_multigrid_preconditioned_solver
  simple_solver --> mixed_multigrid_solver
  iterative_refinement --> mixed_precision_ir
  preconditioned_solver --> multigrid_preconditioned_solver
  multigrid_preconditioned_solver --> multigrid_preconditioned_solver_customized
  simple_solver --> nine_pt_stencil_solver
  three_pt_stencil_solver --> nine_pt_stencil_solver
  poisson_solver --> nine_pt_stencil_solver
  simple_solver_logging --> papi_logging
  simple_solver --> par_ilu_convergence
  simple_solver_logging --> performance_debugging
  minimal_cuda_solver --> performance_debugging
  simple_solver --> poisson_solver
  simple_solver --> preconditioned_solver
  simple_solver --> preconditioner_export
  preconditioned_solver --> reordered_preconditioned_solver
  heat_equation --> schroedinger_splitting
  simple_solver --> simple_solver_logging
  minimal_cuda_solver --> simple_solver_logging
  simple_solver --> three_pt_stencil_solver
  poisson_solver --> three_pt_stencil_solver
  click adaptiveprecision_blockjacobi "adaptiveprecision-blockjacobi.html" "Open adaptiveprecision-blockjacobi"
  click batched_solver "batched-solver.html" "Open batched-solver"
  click cb_gmres "cb-gmres.html" "Open cb-gmres"
  click custom_logger "custom-logger.html" "Open custom-logger"
  click custom_matrix_format "custom-matrix-format.html" "Open custom-matrix-format"
  click custom_stopping_criterion "custom-stopping-criterion.html" "Open custom-stopping-criterion"
  click distributed_multigrid_preconditioned_solver "distributed-multigrid-preconditioned-solver.html" "Open distributed-multigrid-preconditioned-solver"
  click distributed_multigrid_reuse "distributed-multigrid-reuse.html" "Open distributed-multigrid-reuse"
  click distributed_solver "distributed-solver.html" "Open distributed-solver"
  click external_lib_interfacing "external-lib-interfacing.html" "Open external-lib-interfacing"
  click file_config_solver "file-config-solver.html" "Open file-config-solver"
  click ginkgo_overhead "ginkgo-overhead.html" "Open ginkgo-overhead"
  click ginkgo_ranges "ginkgo-ranges.html" "Open ginkgo-ranges"
  click heat_equation "heat-equation.html" "Open heat-equation"
  click ilu_preconditioned_solver "ilu-preconditioned-solver.html" "Open ilu-preconditioned-solver"
  click inverse_iteration "inverse-iteration.html" "Open inverse-iteration"
  click ir_ilu_preconditioned_solver "ir-ilu-preconditioned-solver.html" "Open ir-ilu-preconditioned-solver"
  click iterative_refinement "iterative-refinement.html" "Open iterative-refinement"
  click kokkos_assembly "kokkos-assembly.html" "Open kokkos-assembly"
  click minimal_cuda_solver "minimal-cuda-solver.html" "Open minimal-cuda-solver"
  click mixed_multigrid_preconditioned_solver "mixed-multigrid-preconditioned-solver.html" "Open mixed-multigrid-preconditioned-solver"
  click mixed_multigrid_solver "mixed-multigrid-solver.html" "Open mixed-multigrid-solver"
  click mixed_precision_ir "mixed-precision-ir.html" "Open mixed-precision-ir"
  click mixed_spmv "mixed-spmv.html" "Open mixed-spmv"
  click multigrid_preconditioned_solver "multigrid-preconditioned-solver.html" "Open multigrid-preconditioned-solver"
  click multigrid_preconditioned_solver_customized "multigrid-preconditioned-solver-customized.html" "Open multigrid-preconditioned-solver-customized"
  click nine_pt_stencil_solver "nine-pt-stencil-solver.html" "Open nine-pt-stencil-solver"
  click papi_logging "papi-logging.html" "Open papi-logging"
  click par_ilu_convergence "par-ilu-convergence.html" "Open par-ilu-convergence"
  click performance_debugging "performance-debugging.html" "Open performance-debugging"
  click poisson_solver "poisson-solver.html" "Open poisson-solver"
  click preconditioned_solver "preconditioned-solver.html" "Open preconditioned-solver"
  click preconditioner_export "preconditioner-export.html" "Open preconditioner-export"
  click reordered_preconditioned_solver "reordered-preconditioned-solver.html" "Open reordered-preconditioned-solver"
  click schroedinger_splitting "schroedinger-splitting.html" "Open schroedinger-splitting"
  click simple_solver "simple-solver.html" "Open simple-solver"
  click simple_solver_logging "simple-solver-logging.html" "Open simple-solver-logging"
  click three_pt_stencil_solver "three-pt-stencil-solver.html" "Open three-pt-stencil-solver"
  classDef basic fill:#dcfce7,stroke:#16a34a,color:#14532d
  classDef precond fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
  classDef stop fill:#fef3c7,stroke:#d97706,color:#78350f
  classDef mixed fill:#fce7f3,stroke:#db2777,color:#831843
  classDef logging fill:#ede9fe,stroke:#7c3aed,color:#4c1d95
  classDef tech fill:#f1f5f9,stroke:#475569,color:#1e293b
  classDef batched fill:#ffe4e6,stroke:#e11d48,color:#881337
  classDef distributed fill:#ffedd5,stroke:#ea580c,color:#7c2d12
  class adaptiveprecision_blockjacobi mixed
  class mixed_multigrid_solver mixed
  class mixed_precision_ir mixed
  class mixed_spmv mixed
  class batched_solver batched
  class cb_gmres tech
  class custom_matrix_format tech
  class external_lib_interfacing tech
  class file_config_solver tech
  class ginkgo_overhead tech
  class ginkgo_ranges tech
  class heat_equation tech
  class inverse_iteration tech
  class kokkos_assembly tech
  class schroedinger_splitting tech
  class custom_logger logging
  class papi_logging logging
  class performance_debugging logging
  class simple_solver_logging logging
  class custom_stopping_criterion stop
  class distributed_multigrid_preconditioned_solver distributed
  class distributed_multigrid_reuse distributed
  class distributed_solver distributed
  class ilu_preconditioned_solver precond
  class ir_ilu_preconditioned_solver precond
  class mixed_multigrid_preconditioned_solver precond
  class multigrid_preconditioned_solver precond
  class multigrid_preconditioned_solver_customized precond
  class par_ilu_convergence precond
  class preconditioned_solver precond
  class preconditioner_export precond
  class reordered_preconditioned_solver precond
  class iterative_refinement basic
  class minimal_cuda_solver basic
  class nine_pt_stencil_solver basic
  class poisson_solver basic
  class simple_solver basic
  class three_pt_stencil_solver basic
```

## Browse by category

### Basic solver setups

- [iterative-refinement](iterative-refinement.md): The iterative refinement solver example.
- [minimal-cuda-solver](minimal-cuda-solver.md): The minimal CUDA solver example.
- [nine-pt-stencil-solver](nine-pt-stencil-solver.md): The 9-point stencil example.
- [poisson-solver](poisson-solver.md): The poisson solver example.
- [simple-solver](simple-solver.md): The simple solver example.
- [three-pt-stencil-solver](three-pt-stencil-solver.md): The 3-point stencil example.

### Preconditioners

- [ilu-preconditioned-solver](ilu-preconditioned-solver.md): The ILU-preconditioned solver example.
- [ir-ilu-preconditioned-solver](ir-ilu-preconditioned-solver.md): The IR-ILU preconditioned solver example.
- [mixed-multigrid-preconditioned-solver](mixed-multigrid-preconditioned-solver.md): The preconditioned solver example.
- [multigrid-preconditioned-solver](multigrid-preconditioned-solver.md): The preconditioned solver example.
- [multigrid-preconditioned-solver-customized](multigrid-preconditioned-solver-customized.md): The customized multigrid preconditioned solver example.
- [par-ilu-convergence](par-ilu-convergence.md): The ParILU convergence example.
- [preconditioned-solver](preconditioned-solver.md): The preconditioned solver example.
- [preconditioner-export](preconditioner-export.md): The preconditioner export example.
- [reordered-preconditioned-solver](reordered-preconditioned-solver.md): The reordered preconditioned solver example.

### Stopping criteria

- [custom-stopping-criterion](custom-stopping-criterion.md): The custom stopping criterion creation example.

### Mixed precision

- [adaptiveprecision-blockjacobi](adaptiveprecision-blockjacobi.md): The preconditioned solver example.
- [mixed-multigrid-solver](mixed-multigrid-solver.md): The mixed multigrid solver example.
- [mixed-precision-ir](mixed-precision-ir.md): The Mixed Precision Iterative Refinement (MPIR) solver example.
- [mixed-spmv](mixed-spmv.md): The mixed spmv example.

### Logging and observability

- [custom-logger](custom-logger.md): The simple solver with a custom logger example.
- [papi-logging](papi-logging.md): The papi logging example.
- [performance-debugging](performance-debugging.md): The simple solver with performance debugging example.
- [simple-solver-logging](simple-solver-logging.md): The simple solver with logging example.

### Techniques and integrations

- [cb-gmres](cb-gmres.md): The CB-GMRES solver example.
- [custom-matrix-format](custom-matrix-format.md): The custom matrix format example.
- [external-lib-interfacing](external-lib-interfacing.md): The external library(deal.II) interfacing example.
- [file-config-solver](file-config-solver.md): The file config solver example.
- [ginkgo-overhead](ginkgo-overhead.md): The ginkgo overhead measurement example.
- [ginkgo-ranges](ginkgo-ranges.md): The ranges and accessor example.
- [heat-equation](heat-equation.md): The heat equation example.
- [inverse-iteration](inverse-iteration.md): The inverse iteration example.
- [kokkos-assembly](kokkos-assembly.md): The Kokkos assembly example.
- [schroedinger-splitting](schroedinger-splitting.md): The Schroedinger equation example.

### Batched functionality

- [batched-solver](batched-solver.md): Using and interfacing with a batched solver.

### Distributed solvers

- [distributed-multigrid-preconditioned-solver](distributed-multigrid-preconditioned-solver.md): The distributed multigrid preconditioned solver with customized components example.
- [distributed-multigrid-reuse](distributed-multigrid-reuse.md): The distributed multigrid reuse in a Picard iteration example.
- [distributed-solver](distributed-solver.md): The distributed solver example.

```{toctree}
:maxdepth: 1
:hidden:

adaptiveprecision-blockjacobi
batched-solver
cb-gmres
custom-logger
custom-matrix-format
custom-stopping-criterion
distributed-multigrid-preconditioned-solver
distributed-multigrid-reuse
distributed-solver
external-lib-interfacing
file-config-solver
ginkgo-overhead
ginkgo-ranges
heat-equation
ilu-preconditioned-solver
inverse-iteration
ir-ilu-preconditioned-solver
iterative-refinement
kokkos-assembly
minimal-cuda-solver
mixed-multigrid-preconditioned-solver
mixed-multigrid-solver
mixed-precision-ir
mixed-spmv
multigrid-preconditioned-solver
multigrid-preconditioned-solver-customized
nine-pt-stencil-solver
papi-logging
par-ilu-convergence
performance-debugging
poisson-solver
preconditioned-solver
preconditioner-export
reordered-preconditioned-solver
schroedinger-splitting
simple-solver
simple-solver-logging
three-pt-stencil-solver
```
