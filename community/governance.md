# Governance

Ginkgo is a community-driven open-source project under the
[BSD-3-Clause license](https://github.com/ginkgo-project/ginkgo/blob/develop/LICENSE).
This page documents how the project is structured, how decisions are
made, and where the boundaries are between core maintainers and the
broader contributor base.

## Project structure

Ginkgo is developed by a community spanning academic institutions,
national laboratories, and individual contributors. The full list of
people who have committed code is recorded in
[`contributors.txt`](https://github.com/ginkgo-project/ginkgo/blob/develop/contributors.txt)
in the source tree — every entry there has explicitly placed their
contributions under the project's BSD-3-Clause licence. The live
contributor graph (with activity over time) lives at
[github.com/ginkgo-project/ginkgo/graphs/contributors](https://github.com/ginkgo-project/ginkgo/graphs/contributors).

## Decision-making

Day-to-day technical decisions are made by the **core maintainer team** —
the set of project members with merge rights on the `ginkgo-project/ginkgo`
repository. Disagreements are resolved through public discussion on the
PR, in
[GitHub Discussions](https://github.com/ginkgo-project/ginkgo/discussions),
or by direct conversation in maintainer meetings.

Larger strategic decisions (release timing, deprecation policy, support
matrix for new compilers / hardware) are coordinated across the
participating institutions and discussed openly with the community before
they land.

There is currently **no formal RFC process** — proposals for substantial
new features typically start as a GitHub issue or discussion thread, get
iterated on in conversation, and then move to a draft PR. If you have a
design that's bigger than a single PR, opening a discussion first is
encouraged.

## Pull request approval

The merge policy is documented in detail on
[Submit a pull request](../user-guide/how-to/contributing/submit-a-pr.md).
The condensed version:

- All PRs target `develop`. `main` only moves at release time.
- **One approval** from a project member is required before a PR can be
  merged. Substantial changes typically attract more reviewers
  organically.
- CI must be green on both GitHub Actions (the cross-platform matrix) and
  the mirrored GitLab pipeline (the full Linux × backend matrix) before
  merge.
- Merges are **rebase-only** — `develop` has a linear history by policy.

## Code of conduct

Ginkgo expects everyone interacting with the project — in issues, PRs,
discussions, email, or in person — to be welcoming, respectful, and
constructive. The project follows the spirit of the
[Contributor Covenant](https://www.contributor-covenant.org/). If you
witness or experience behaviour that violates that standard, contact the
maintainers at
[ginkgo.library@gmail.com](mailto:ginkgo.library@gmail.com); reports will
be handled confidentially.

## Releases

Releases are tagged on the `main` branch. `develop` continuously receives
new work; periodically `develop` is merged into `main` and a versioned
release is cut.

- Tagged releases are visible on the
  [GitHub Releases page](https://github.com/ginkgo-project/ginkgo/releases).
- Release tags follow `vMAJOR.MINOR.PATCH`. A breaking-change bump moves
  `MAJOR`; new features bump `MINOR`; bug-fix-only releases bump `PATCH`.
- The CHANGELOG is maintained on the
  [project wiki](https://github.com/ginkgo-project/ginkgo/wiki/Changelog)
  — every user-visible PR adds a line referencing back to the PR number.
- ABI compatibility against the last release is checked on every PR by
  the `abidiff` GitHub Action.

## License

Ginkgo itself is distributed under the
[BSD-3-Clause license](https://github.com/ginkgo-project/ginkgo/blob/develop/LICENSE).
Third-party software pulled in via optional build options carries its own
licence; see
[`ABOUT-LICENSING.md`](https://github.com/ginkgo-project/ginkgo/blob/develop/ABOUT-LICENSING.md)
for the breakdown. The project participates in the
[REUSE specification](https://reuse.software/), so every file in the
source tree carries explicit `SPDX-License-Identifier` metadata.

## Funding and acknowledgement

Ginkgo's development has been supported over the years by, among others:

- The **Helmholtz Association** (HIDA programme, KIT institutional
  funding).
- The **US Department of Energy** via the
  [Exascale Computing Project (ECP)](https://www.exascaleproject.org/).
- The **xSDK** ecosystem of math libraries.
- Cross-cutting open-source-research funding from the participating
  universities.

If you use Ginkgo in academic work, please cite — see
[Citing Ginkgo](citing.md). Citations and concrete usage stories are
both essential to securing the continued funding that pays for ongoing
development.

## Get in touch

- [GitHub Issues](https://github.com/ginkgo-project/ginkgo/issues) — bugs
  and feature requests.
- [GitHub Discussions](https://github.com/ginkgo-project/ginkgo/discussions)
  — questions, design discussions, broader conversation.
- [`ginkgo.library@gmail.com`](mailto:ginkgo.library@gmail.com) — direct
  contact with the maintainer team.
