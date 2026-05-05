# Explanation

These pages explain Ginkgo's design and architecture. Pick them when you want
to *understand* — for "how do I do X", see the How-To guides; for "walk me
through a worked example", see the Tutorials.

## Foundation

```{toctree}
:maxdepth: 1
executor-model
linop-and-composition
memory-ownership
views-zero-copy
```

## Matrix data and formats

```{toctree}
:maxdepth: 1
matrix-data
matrix-formats
formats/csr
formats/coo
formats/ell
formats/sellp
formats/hybrid
formats/fbcsr
```

## Solvers, preconditioners, criteria

```{toctree}
:maxdepth: 1
solvers
preconditioners
stopping-criteria
mixed-precision
```

## Distributed

```{toctree}
:maxdepth: 1
distributed
distributed/partition-and-index-map
distributed/communication-primitives
distributed/solvers-and-preconditioners
distributed/neighborhood-communicators
```

## Batched

```{toctree}
:maxdepth: 1
batched
batched/per-solver-design
```

## Cross-cutting features

```{toctree}
:maxdepth: 1
logging
config-files
```
