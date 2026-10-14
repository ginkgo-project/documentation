# documentation

Ginkgo documentation pages, published at <https://gko-project.org/>.

## Versions and branches

Each docs branch documents one Ginkgo version:

| Docs branch    | Documents Ginkgo | Published at                              |
|----------------|------------------|-------------------------------------------|
| `develop`      | `develop`        | <https://gko-project.org/> (default)      |
| `main`         | `v1.11.0`        | <https://gko-project.org/1.11/> (stable)  |
| `release/1.10` | `v1.10.0`        | <https://gko-project.org/1.10/>           |

The Ginkgo ref a branch documents is pinned in its `.ginkgo-ref` file. The
list of published versions, which also feeds the version switcher, is
`_static/switcher.json` on `main`; edit it only there.

The site root serves the develop docs. Their header carries a permanent
orange banner (set in `conf.py`, styled in `_static/ginkgo-theme/custom.css`)
that marks them as unstable and links to the stable docs, the `preferred`
entry of the version list. Older versions show the theme's version warning
instead. A page that only the stable docs have gets a redirect stub at the
root (`scripts/assemble_site.sh`), so its old URL keeps working.

Where changes go:

- Docs for new Ginkgo features → pull request into `develop`.
- Corrections for the current release → pull request into `main`, then
  cherry-pick into `develop` if the text applies there too. Never merge
  `main` into `develop`: it would carry release-only removals into develop.
- Corrections for an older release → pull request into its `release/*`
  branch.
- Shared pages (homepage news, community, citing) → edit on `main`, then
  cherry-pick into the other branches.
- Build and site infrastructure (workflows, `scripts/`, `conf.py`,
  `Makefile`, `_doxygen/`, theme files) → change on `develop` and `main`
  alike, then cherry-pick into `release/*`. At a release `main` takes
  develop's tree, so a fix made only on `main` would be lost.

## Building locally

```bash
make install          # one-time: create .venv and install Python deps
make doxygen-xml      # Doxygen XML for the API reference (see below)
make html             # build the site into _build/html
```

Other useful targets: `make strict` (warnings → errors, as in the PR
check), `make linkcheck`, `make clean`.

The build reads Ginkgo from a sibling checkout, `../upstream`: Doxygen
parses its headers for the API reference, and the example pages include
their full programs from `../upstream/examples/`. Check that checkout out
at the ref in `.ginkgo-ref`:

```bash
git -C ../upstream checkout "$(cat .ginkgo-ref)"
```

`make doxygen-xml` warns when `../upstream` is on a different ref or has
uncommitted changes. `GINKGO_SOURCE_DIR=/path/to/ginkgo` points Doxygen at
another checkout; the example pages still read `../upstream`.

The version highlighted in the switcher comes from `.ginkgo-ref`
(`v1.11.0` → `1.11`); set `GINKGO_DOC_VERSION` to override it.

## Breathe / Doxygen XML

Pages that embed C++ API entities use [Breathe](https://breathe.readthedocs.io/)
directives such as `{doxygenclass}` and `{doxygenfunction}`. They need the
Doxygen XML that `make doxygen-xml` writes to `_doxygen/xml/` (gitignored)
from the Ginkgo checkout described above.

The Doxyfile at `_doxygen/Doxyfile` emits XML only. `_doxygen/fix_docstrings.py`
is a Doxygen input filter that repairs a few doc comments in released
Ginkgo headers that Doxygen would otherwise misparse; on current headers it
changes nothing.

If `_doxygen/xml/` is missing, `make html` still succeeds; only pages
that contain Breathe directives emit warnings about unresolved entities.
A self-contained smoke-test page lives at `_doxygen/smoke-test.md`
(orphan, not in any toctree) and renders only when the XML is present.

## Publishing

`.github/workflows/build-and-deploy.yml` builds every version listed in
`_static/switcher.json`, each from its docs branch against its pinned Ginkgo
ref, checks each build with `scripts/check_build.py`, and deploys them
together to GitHub Pages. It runs on every push to `main`, weekly (so
the develop docs at the root follow Ginkgo's develop branch), and whenever
`trigger-deploy.yml` dispatches it after a push to `develop` or
`release/*`. Deploys always run from `main`. If any version fails to
build, nothing is deployed and the previous site stays live.

Pull requests into any version branch run `pr-build.yml`: a strict build
against the pinned Ginkgo ref of the pull request's branch.

## Cutting a docs release

When a Ginkgo release ships (shown for 1.12):

1. Create `release/1.11` from `main`. Its `.ginkgo-ref` stays `v1.11.0`.
2. Promote `develop` to `main`. Do not use a plain `git merge develop`:
   removals made only on `main` (such as release-only page cuts) would be
   carried into the result and delete content that develop still has.
   Instead record a merge whose tree is exactly develop's, keeping
   `main`'s version list:

   ```bash
   git switch main
   git merge -s ours --no-commit develop
   git read-tree -u --reset develop
   git checkout HEAD -- _static/switcher.json
   echo v1.12.0 > .ginkgo-ref
   ```

3. Edit `_static/switcher.json`: add
   `{"name": "1.12 (stable)", "version": "1.12", "url": "https://gko-project.org/1.12/", "branch": "main", "preferred": true}`;
   change the 1.11 entry to `"name": "1.11"` and `"branch": "release/1.11"`
   without `preferred` (its URL stays `https://gko-project.org/1.11/`); drop
   the oldest entry if it is no longer maintained. The develop entry stays at
   the root. Then commit the merge together with the new
   pin and list: `git add .ginkgo-ref _static/switcher.json && git commit`.
4. Push `release/1.11` and `main` together:
   `git push --atomic origin main release/1.11`.

A patch release (e.g. 1.11.1) only changes `main`'s `.ginkgo-ref`.
