#!/usr/bin/env python3
"""Sanity checks on a built documentation site.

Run after `sphinx-build -b html`. Exits non-zero with one `error:` line per
problem, so CI fails before publishing a broken version:

- the version switcher is wired to the production `switcher.json` and knows
  which version this build is;
- every example page's "The plain program" section contains code (it is
  empty when the Ginkgo checkout is missing from `../upstream`);
- the `_static/switcher.json` shipped with the build is well formed.

`--switcher FILE` validates only a version list, before anything is built.

Usage: check_build.py <html-dir> [--version VERSION]
       check_build.py --switcher _static/switcher.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

SITE = "https://gko-project.org/"
SWITCHER_URL = SITE + "_static/switcher.json"
PLAIN_PROGRAM = re.compile(r'<section id="the-plain-program">(.*?)</section>', re.S)
VERSION_MATCH = re.compile(r"theme_switcher_version_match = '([^']*)'")
SPHINX_VERSION = re.compile(r"VERSION: '([^']*)'")


def check_switcher_wiring(html_dir, version):
    index_path = html_dir / "index.html"
    if not index_path.is_file():
        return [f"{index_path}: missing"]
    index = index_path.read_text(encoding="utf-8")
    errors = []
    if f"theme_switcher_json_url = '{SWITCHER_URL}'" not in index:
        errors.append(f"index.html: switcher json_url is not {SWITCHER_URL}")
    match = VERSION_MATCH.search(index)
    if not match or not match.group(1):
        errors.append("index.html: switcher version_match missing or empty")
    else:
        if version is not None and match.group(1) != version:
            errors.append(
                f"index.html: switcher version_match is {match.group(1)!r}, "
                f"expected {version!r}"
            )
        # The theme's version-warning banner compares Sphinx's own
        # `version` (not version_match) against the preferred entry, so
        # the two must agree or every build shows an "unknown version"
        # banner.
        options = html_dir / "_static" / "documentation_options.js"
        if options.is_file():
            sphinx = SPHINX_VERSION.search(options.read_text(encoding="utf-8"))
            sphinx_version = sphinx.group(1) if sphinx else None
            if sphinx_version != match.group(1):
                errors.append(
                    f"_static/documentation_options.js: Sphinx version is "
                    f"{sphinx_version!r}, expected the switcher version "
                    f"{match.group(1)!r}"
                )
    if "ginkgo-version-switcher" in index:
        errors.append("index.html: legacy hand-written version switcher present")
    return errors


def check_plain_programs(html_dir):
    errors = []
    for page in sorted((html_dir / "examples").glob("*.html")):
        match = PLAIN_PROGRAM.search(page.read_text(encoding="utf-8"))
        if match and 'class="highlight' not in match.group(1):
            errors.append(
                f"{page.relative_to(html_dir)}: 'The plain program' has no code"
            )
    return errors


def check_switcher_json(path, name):
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{name}: invalid JSON ({exc})"]
    if not isinstance(entries, list) or not entries:
        return [f"{name}: expected a non-empty list of versions"]
    errors = []
    preferred = [e for e in entries if e.get("preferred")]
    if len(preferred) != 1:
        errors.append(
            f"{name}: expected exactly one preferred entry, found {len(preferred)}"
        )
    for i, entry in enumerate(entries):
        for key in ("name", "version", "url", "branch"):
            if not entry.get(key):
                errors.append(f"{name}: entry {i} has no {key!r}")
        url = entry.get("url", "")
        if not (url.startswith(SITE) and url.endswith("/")):
            errors.append(
                f"{name}: entry {i} url {url!r} must start with {SITE} and end with '/'"
            )
    for key, plural in (("version", "versions"), ("url", "urls"), ("branch", "branches")):
        values = [e.get(key) for e in entries]
        if len(set(values)) != len(values):
            errors.append(f"{name}: duplicate {plural} in {values}")
    # The root serves `main`'s build, and the deploy publishes `main`'s
    # copy of this list; a release edit that leaves the root to another
    # branch (or to nobody) would publish a stale list or no homepage.
    root = [e for e in entries if e.get("url") == SITE]
    if len(root) != 1:
        errors.append(f"{name}: expected exactly one entry at {SITE}, found {len(root)}")
    else:
        if root[0].get("branch") != "main":
            errors.append(f"{name}: the entry at {SITE} must have branch 'main'")
        if not root[0].get("preferred"):
            errors.append(f"{name}: the entry at {SITE} must be preferred")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("html_dir", type=Path, nargs="?")
    parser.add_argument("--version", help="expected switcher version_match")
    parser.add_argument(
        "--switcher", type=Path, metavar="FILE",
        help="only validate this switcher.json (no built site needed)",
    )
    args = parser.parse_args()
    if args.switcher is not None:
        errors = check_switcher_json(args.switcher, str(args.switcher))
    elif args.html_dir is not None:
        shipped = args.html_dir / "_static" / "switcher.json"
        errors = (
            check_switcher_wiring(args.html_dir, args.version)
            + check_plain_programs(args.html_dir)
            + (check_switcher_json(shipped, "_static/switcher.json")
               if shipped.is_file() else [])
        )
    else:
        parser.error("give a built html_dir or --switcher FILE")
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
