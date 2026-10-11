#!/usr/bin/env bash
# Place each version's HTML at its URL path inside the Pages site and add
# main's version list and the CNAME. Fails if any version listed in
# $VERSIONS has no build, or if nothing lands at the site root, so a
# partial site is never published.
# Usage (from the repo root):
#   VERSIONS="$(scripts/versions_matrix.sh _static/switcher.json)" \
#     scripts/assemble_site.sh <artifacts-dir> <site-dir>
# <artifacts-dir> holds one `site-<version>/` directory per version.
set -euo pipefail
artifacts=$1
site=$2
mkdir -p "$site"
while IFS=$'\t' read -r version path; do
    build="$artifacts/site-$version"
    if [ ! -f "$build/index.html" ]; then
        echo "error: no build for version $version in $build" >&2
        exit 1
    fi
    mkdir -p "$site/$path"
    cp -a "$build/." "$site/$path/"
done < <(jq -r '.[] | [.version, .path] | @tsv' <<<"$VERSIONS")
if [ ! -f "$site/index.html" ]; then
    echo "error: no version is published at the site root" >&2
    exit 1
fi
# Every version's switcher reads the list served at the root; serve the
# one this deploy was planned from (main's), whatever the root build held.
mkdir -p "$site/_static"
cp _static/switcher.json "$site/_static/switcher.json"
cp CNAME "$site/CNAME"
