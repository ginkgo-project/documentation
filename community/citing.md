# Citing Ginkgo

If you use Ginkgo in academic work or in any publication, please cite it.
Citations are the most direct way to support continued development —
funding agencies look at them, and they help us justify the time we
spend making the library faster and more capable.

## Primary citation

The canonical Ginkgo reference is the
[2022 ACM TOMS paper](https://dl.acm.org/doi/10.1145/3480935):

```bibtex
@article{ginkgo-toms-2022,
  title   = {{Ginkgo: A Modern Linear Operator Algebra Framework for High Performance Computing}},
  author  = {Anzt, Hartwig and Cojean, Terry and Flegar, Goran and G\"{o}bel, Fritz and
             Gr\"{u}tzmacher, Thomas and Nayak, Pratik and Ribizel, Tobias and
             Tsai, Yuhsiang Mike and Quintana-Ort{\'\i}, Enrique S.},
  journal = {ACM Transactions on Mathematical Software},
  volume  = {48},
  number  = {1},
  year    = {2022},
  pages   = {1--33},
  doi     = {10.1145/3480935}
}
```

If you can only fit one citation in your paper, use that one — it covers
the library's overall design and serves as the default attribution.

## Software citation (JOSS)

A
[Journal of Open Source Software](https://joss.theoj.org/papers/10.21105/joss.02260)
paper accompanies the software and is useful when you want to cite Ginkgo
*as software* rather than the underlying methodology:

```bibtex
@article{ginkgo-joss-2020,
  title   = {{Ginkgo: A High Performance Numerical Linear Algebra Library}},
  author  = {Anzt, Hartwig and Cojean, Terry and Yen-Chen, Chen and Flegar, Goran and
             G\"{o}bel, Fritz and Gr\"{u}tzmacher, Thomas and Nayak, Pratik and
             Ribizel, Tobias and Tsai, Yuhsiang Mike},
  journal = {Journal of Open Source Software},
  volume  = {5},
  number  = {52},
  pages   = {2260},
  year    = {2020},
  doi     = {10.21105/joss.02260}
}
```

## Citing a specific feature

Several Ginkgo subsystems have their own peer-reviewed publications. If
your work relies heavily on a particular feature, citing both the
flagship TOMS paper *and* the subsystem paper is the most accurate
attribution.

The relevant publications are listed at the bottom of the corresponding
concept page in the user guide:

- **Distributed solvers** —
  [Distributed computing → Publications](../user-guide/concepts/distributed.md)
  (also covers Schwarz, distributed Pgm).
- **Batched solvers** —
  [Batched functionality → Publications](../user-guide/concepts/batched.md).
- **Mixed-precision techniques** —
  [Mixed precision → Publications](../user-guide/concepts/mixed-precision.md).
- **Preconditioners** (block Jacobi, ParILU, ParILUT, AMG) —
  [Preconditioners → Publications](../user-guide/concepts/preconditioners.md).
- **Hardware-portable kernels and the executor model** — the TOMS paper
  above covers this comprehensively.

Each page links the BibTeX entries directly, so you can copy them from
there into your bibliography.

## Citing a specific release

Tagged releases on
[GitHub](https://github.com/ginkgo-project/ginkgo/releases) carry the
version number and a date. For reproducibility-critical work, cite the
exact version you used in addition to the paper:

```text
Ginkgo X.Y.Z, https://github.com/ginkgo-project/ginkgo, tag vX.Y.Z, accessed YYYY-MM-DD.
```

If you need a DOI for a specific release for archival purposes,
maintainers can mint one on
[Zenodo](https://zenodo.org/) on request — ask in
[Discussions](https://github.com/ginkgo-project/ginkgo/discussions) or by
email.

## Crediting individual authors

The full list of authors who have contributed to Ginkgo is in the
[`contributors.txt`](https://github.com/ginkgo-project/ginkgo/blob/develop/contributors.txt)
file. If you are writing a software-focused paper (rather than a method
paper) and want to credit specific contributors beyond the citation
list, the maintainer team is happy to advise — open a discussion or send
an email.

## Telling us about your work

If you publish a paper or thesis that uses Ginkgo, we'd love to hear
about it — it helps us understand which features are landing in
practice, and it directly supports funding renewals. Drop a line in
[Discussions](https://github.com/ginkgo-project/ginkgo/discussions) or
email
[ginkgo.library@gmail.com](mailto:ginkgo.library@gmail.com) with a link
to the paper.
