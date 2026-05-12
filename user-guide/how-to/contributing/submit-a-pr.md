# Submit a pull request

Ginkgo's contribution flow is GitHub-based, but the heavy CI is mirrored
to GitLab. A PR has to clear both pipelines plus formatting / lint /
spell-check checks before two maintainers can approve and merge it onto
the `develop` branch.

## Branch and base

- All pull requests target `develop`. `main` only receives merges
  from `develop` at release time and is tagged for each release. Do
  not open PRs against `main`.
- Rebase onto `develop` before requesting the final review. Ginkgo's
  history is linear by policy — merges are produced via rebase, not via
  merge commits.
- Push your branch to your fork (or to a branch on the public mirror if
  you have write access). The full CI matrix runs when you open the PR.

## Commits

The commit-message style from `CONTRIBUTING.md` is:

```
Summary of the changes in a sentence, max 50 chars.

More detailed comments:
+ Changes that have been added.
- Changes that have been removed.

Related PR: https://github.com/ginkgo-project/ginkgo/pull/<PR-number>
```

The first line is a verb-led summary at most 50 characters. Empty line.
Then a description that walks through what changed and why — not how;
the diff already shows that.

If you took an idea from a reviewer comment, attribute it with
`Co-authored-by`:

```
Suggested by reviewer.

Co-authored-by: Name <email@domain>
```

Each commit must pass `pre-commit run`. With `GINKGO_DEVEL_TOOLS=ON`
this is already installed as a git hook. From the build directory,
`make format` runs the same hooks across staged files.

## What CI runs

Two pipelines run on every push to the PR branch.

**GitHub Actions** runs the portable checks:

| Workflow | Checks |
|---|---|
| `check-formatting.yml` | `pre-commit` (clang-format v14, end-of-file fixer, cmake-format, REUSE annotations). |
| `osx.yml` | Build + test on macOS. |
| `windows-msvc-cuda.yml`, `windows-msvc-mpi.yml` | Build + test on Windows / MSVC. |
| `arm.yml` | Build + test on ARM. |
| `intel.yml` | Build + test on Intel-toolchain (DPC++/SYCL). |
| `spell-check.yml` | `typos`-based spell-check. |
| `abidiff.yml` | ABI compatibility against the last release. |

**GitLab CI** (mirrored from the public repo) runs the comprehensive
build/test matrix. Stages are:

| Stage | Jobs |
|---|---|
| `build` | One job per `(CUDA version × MPI variant × compiler × build type × shared/static)` row. Current matrix covers CUDA 11.0–13.1, ROCm 6.2.0–7.2.0, no-CUDA OMP-only, and no-GPU configurations. Includes a `glibcxx-debug-release` row that enables `_GLIBCXX_DEBUG`. |
| `test` | Runs the unit-test suite on each backend produced in `build`. |
| `code_quality` | Static analysis (clang-tidy, IWYU on a subset). |
| `deploy` | Documentation deployment from `develop`. |
| `QoS_tools` | Optional quality-of-service tooling. |
| `benchmark-*` | The benchmark drivers exercised on a smoke subset of matrices, one stage per executor. |

A PR has to be green across both pipelines. Treat a red GitHub Actions
build (formatting, MacOS, Windows, ARM, Intel) the same way as a red
GitLab build — re-push, don't merge over a failure.

## Labels

The label bot auto-applies path-derived labels when you open the PR
(`mod:core`, `mod:cuda`, `reg:benchmarking`, `type:solver`, ...). Add
the missing semantic labels yourself:

- `1:ST:ready-for-review` when you want others to review.
- `1:ST:WIP` while you're still iterating.
- `1:ST:no-changelog-entry` if your change genuinely doesn't belong in
  the wiki changelog (formatting-only PRs, internal refactors with no
  user-visible effect). Without this label, the CI requires a wiki
  changelog entry pointing back to the PR — see the next section.

The full label vocabulary is visible in
[`.github/label.sh`](https://github.com/ginkgo-project/ginkgo/blob/develop/.github/label.sh).

## Changelog entry

User-facing changes need an entry in the [wiki
Changelog](https://github.com/ginkgo-project/ginkgo/wiki/Changelog).
The check is automated: `.github/check-wiki-changelog.sh` looks for a
link of the form `[#1234](https://github.com/ginkgo-project/ginkgo/pull/1234)`
in the wiki page. If the line is missing, the check fails and the PR
can't be marked ready for merge.

Two ways to satisfy the check:

1. Add the entry to the wiki Changelog yourself. The wiki is
   editable by anyone with a GitHub account.
2. Apply the `1:ST:no-changelog-entry` label and the check is skipped.
   Use this sparingly — formatting-only PRs, internal refactors with
   no user-visible API or behaviour change.

Release branches (`release-*`) skip the check automatically.

## Review and merge

- **Two approvals** from project maintainers are required before merge.
- The assignee is the person responsible for the merge — currently this
  must be a member of the `ginkgo-project` organisation, because GitHub
  doesn't allow assigning users outside the org.
- Once the PR has two approvals and the CI is green, a maintainer marks
  it `READY TO MERGE`. At that point you (the PR author) rebase onto
  the current `develop` if needed and force-push the result.
- Merges land via rebase. There are no merge commits on `develop`.

## A typical PR walkthrough

1. Fork, branch off `develop`, make changes.
2. Configure with `GINKGO_DEVEL_TOOLS=ON` (see [Set up a development
   environment](setup-dev-environment.md)) so the pre-commit hook
   formats every commit.
3. Commit in small logical units. Each commit message is a 50-character
   summary plus a body. Run `make format` if you skipped the hook.
4. Run a fast local check: `make quick_test`.
5. Push to your fork. Open the PR against `develop`. The label bot
   adds path labels; you add `1:ST:ready-for-review` (or `1:ST:WIP`)
   and the changelog-skip label if appropriate.
6. Add a wiki-changelog entry if the change is user-visible.
7. CI runs. Iterate on failures by pushing fixes — don't amend the
   review-history commits, push new commits.
8. Address review comments. Maintainers can suggest changes
   through GitHub's suggestion blocks; accept them or reply.
9. Once approved and green, rebase onto current `develop`, force-push,
   and one of the assignees performs the rebase-merge.

:::{seealso}
- `CONTRIBUTING.md` in the Ginkgo source tree — coding style,
  whitespace conventions, naming.
- [Set up a development environment](setup-dev-environment.md) — get
  the pre-commit hook running before your first commit.
- [Write tests](write-tests.md) — what reviewers expect to see for any
  behavioural change.
- [Run the benchmark suite](run-benchmark-suite.md) — relevant when a
  PR can affect performance.
:::
