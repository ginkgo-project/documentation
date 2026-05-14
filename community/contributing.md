# Contributing to Ginkgo

Ginkgo is open source under the [BSD-3-Clause license](https://github.com/ginkgo-project/ginkgo/blob/develop/LICENSE)
and welcomes contributions from anyone — academic, industrial, or hobbyist.
This page is the orientation for new contributors. If you're looking for
detailed mechanics (kernel registration, test patterns, CI matrix), every
section here links into the deeper how-to pages in the
[Contributing to Ginkgo](../user-guide/how-to/contributing/index.md) section
of the user guide.

The canonical contribution document in the source tree is
[`CONTRIBUTING.md`](https://github.com/ginkgo-project/ginkgo/blob/develop/CONTRIBUTING.md).
This page is a friendlier overview; the user-guide pages are the polished,
versioned how-tos; `CONTRIBUTING.md` is the always-authoritative reference.

## Ways to contribute

You don't need to write C++ to help Ginkgo.

- **Report a bug** or unexpected behaviour — open an issue, give us a
  small reproducer, and tell us what backend you're on. See
  [Reporting issues](#reporting-issues-and-asking-questions).
- **Ask a question** in
  [GitHub Discussions](https://github.com/ginkgo-project/ginkgo/discussions).
  Discussions are also where to bring up ideas before turning them into a
  feature request.
- **Improve documentation.** These pages live in the
  [ginkgo-project/documentation](https://github.com/ginkgo-project/documentation)
  repository; the API reference lives in the main source tree as Doxygen
  comments. Both accept PRs.
- **Add or improve an example.** The
  [examples tree](https://github.com/ginkgo-project/ginkgo/tree/develop/examples)
  is the user-facing how-to layer; new examples are warmly received. See
  [Examples](../examples/index.md) for the catalogue.
- **Review a pull request.** Reviewer bandwidth is the single biggest
  constraint on Ginkgo's release cadence; thoughtful reviews from outside
  the core team help a lot.
- **Add a feature.** Solvers, preconditioners, kernels, format conversions,
  configuration sugar — see the per-topic guides linked below.
- **Cite Ginkgo** in your papers (see
  [Citing Ginkgo](citing.md)) — citations help us secure continued funding
  and motivate ongoing development.

## Quick-start checklist

For a code change, the minimum path is:

1. **Fork** [`ginkgo-project/ginkgo`](https://github.com/ginkgo-project/ginkgo)
   and clone your fork.
2. **Configure with development tools** so the pre-commit hook is wired in
   automatically:
   ```sh
   cmake -B build -DGINKGO_DEVEL_TOOLS=ON
   cmake --build build
   ```
   This installs the
   [pre-commit](https://pre-commit.com/) hook that runs `clang-format`,
   `gersemi` (CMake), `typos`, and the REUSE annotator on every commit.
   See [Set up a development environment](../user-guide/how-to/contributing/setup-dev-environment.md)
   for the full setup.
3. **Branch off `develop`** — all PRs target `develop`, never `main`.
   `main` only receives merges at release time.
4. **Make your changes**, run a fast local check with
   `make quick_test` (from the build directory).
5. **Write tests** — every behavioural change needs them.
   [Write tests](../user-guide/how-to/contributing/write-tests.md) covers
   the test tree layout and the `CommonTestFixture` patterns.
6. **Commit in small logical chunks.** The pre-commit hook formats every
   commit. See [Submit a pull request — commits](../user-guide/how-to/contributing/submit-a-pr.md#commits)
   for the message style.
7. **Push to your fork** and open a PR against `develop`. The label bot
   adds module/area labels automatically; you add the semantic state label
   (`1:ST:ready-for-review` or `1:ST:WIP`).
8. **Iterate on review comments.** Push new commits — don't amend the
   review history. Once approved and CI is green, a project member marks
   it `READY TO MERGE` and the rebase-merge follows.

The full play-by-play is on
[Submit a pull request](../user-guide/how-to/contributing/submit-a-pr.md).

## What kind of change are you making?

Each of these has a dedicated walk-through with the exact files to touch,
the macros to use, the test pattern, and the CMake glue:

- **Add a kernel** to an existing class —
  [Add a new kernel](../user-guide/how-to/contributing/add-a-kernel.md).
- **Add a matrix format** —
  [Add a new matrix format](../user-guide/how-to/contributing/add-a-matrix-format.md).
- **Add a solver** —
  [Add a new solver](../user-guide/how-to/contributing/add-a-solver.md).
- **Add a preconditioner** —
  [Add a new preconditioner](../user-guide/how-to/contributing/add-a-preconditioner.md).
- **Add a backend extension** (vendor-library wrapper, Kokkos bridge, ...)
  — [Add a backend extension](../user-guide/how-to/contributing/add-a-backend-extension.md).
- **Improve build / iteration time** —
  [Speed up rebuilds](../user-guide/how-to/contributing/speed-up-builds.md).
- **Touch benchmarks** —
  [Run the benchmark suite](../user-guide/how-to/contributing/run-benchmark-suite.md).

## Coding standards

Most of what `CONTRIBUTING.md` codifies is enforced by the pre-commit hook
— if your commits pass `pre-commit run`, you're 90 % of the way there. The
remaining 10 % is convention:

- **Naming.** Files / functions / variables / namespaces use `snake_case`;
  polymorphic classes use `CamelCase`; macros use `CAPITAL_CASE` and must
  start with `GKO_`; non-public members end with `_`. Type template
  parameters are `CamelCase` (`ValueType`), non-type ones are `snake_case`
  (`subwarp_size`).
- **No single-line control flow.** Always brace `if` / `for` / `while`.
- **One variable per declaration.** `int* x, y;` is not allowed (the
  first is a pointer, the second isn't — confusing).
- **No circular dependencies.** Polymorphic classes cannot be
  *instantiated* inside a kernel module (`cuda/`, `hip/`, `omp/`,
  `dpcpp/`, `reference/`). Calling another kernel by name is fine. The
  `no-circular-deps` CI check enforces this; you can run it locally with
  `-DGINKGO_CHECK_CIRCULAR_DEPS=ON` in CMake.
- **Tests follow the AAA pattern** (Arrange / Act / Assert), one blank
  line between sections.

For the rest — include-statement grouping, whitespace within structures,
control-flow nuances — the
[`CONTRIBUTING.md`](https://github.com/ginkgo-project/ginkgo/blob/develop/CONTRIBUTING.md)
file in the source tree is the reference. The pre-commit hook handles most
of it automatically.

## Reporting issues and asking questions

- **Bugs / feature requests** —
  [open an issue](https://github.com/ginkgo-project/ginkgo/issues/new).
  Useful issue contents: a minimal reproducer, the Ginkgo version (`git
  rev-parse HEAD` is fine), the executor / backend you saw the problem
  on, and the full output of any error / assertion message.
- **Usage questions, design discussion** —
  [GitHub Discussions](https://github.com/ginkgo-project/ginkgo/discussions).
  Discussions are a better fit than issues for "is this the right way to
  …?" and "how do I do X with Ginkgo?".
- **Direct email** — for things that don't fit either of the above,
  reach the maintainers at
  [ginkgo.library@gmail.com](mailto:ginkgo.library@gmail.com).

## Attribution

If a reviewer suggests an idea or change you adopt, attribute them in the
commit:

```text
Commit message describing the change.

Co-authored-by: Name <email@domain>
```

Co-authored commits show up in GitHub with both authors' avatars and
contribute to both contributors' graphs. By signing your contributions you
also place them under the
[BSD-3-Clause license](https://github.com/ginkgo-project/ginkgo/blob/develop/LICENSE)
— see
[`contributors.txt`](https://github.com/ginkgo-project/ginkgo/blob/develop/contributors.txt)
for the running list.

## What happens after I open a PR?

The condensed version (full play-by-play in
[Submit a pull request](../user-guide/how-to/contributing/submit-a-pr.md)):

1. **CI runs** — GitHub Actions covers macOS / Windows / ARM / Intel /
   formatting / spell-check / ABI; the mirrored GitLab pipeline covers the
   comprehensive Linux build × backend matrix and the benchmark smoke
   suite.
2. **A maintainer or community reviewer takes a look.** They may suggest
   changes through GitHub's suggestion-block UI or in inline comments.
3. **You iterate.** Push fix commits (don't amend the review history).
4. **Once approved and green**, a project member marks the PR
   `READY TO MERGE`. The PR author rebases onto current `develop` and
   force-pushes. The merge lands via rebase — Ginkgo's `develop` history
   is linear by policy.

## See also

- [`CONTRIBUTING.md`](https://github.com/ginkgo-project/ginkgo/blob/develop/CONTRIBUTING.md)
  in the Ginkgo source tree — coding style, whitespace conventions,
  naming, all the corners not surfaced here.
- [Set up a development environment](../user-guide/how-to/contributing/setup-dev-environment.md).
- [Submit a pull request](../user-guide/how-to/contributing/submit-a-pr.md).
- [Write tests](../user-guide/how-to/contributing/write-tests.md).
- [Governance](governance.md) — how decisions are made and who maintains
  what.
- [Citing Ginkgo](citing.md) — citations from your work also help.
