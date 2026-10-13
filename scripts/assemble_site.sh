#!/usr/bin/env bash
# Place each version's HTML at its URL path inside the Pages site, add
# redirect stubs at the root for pages only the stable docs have, and add
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
# The root serves the develop docs. A page that only the stable docs have
# (an API develop has removed, say) keeps its root URL: a stub there
# redirects to the stable copy.
stable_url=$(jq -r 'first.stable_url' <<<"$VERSIONS")
stable_path=${stable_url#https://gko-project.org/}
stable_path=${stable_path%/}
if [ -n "$stable_path" ]; then
    while IFS= read -r page; do
        page=${page#./}
        [ -e "$site/$page" ] && continue
        mkdir -p "$(dirname "$site/$page")"
        cat > "$site/$page" <<EOF
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Ginkgo stable documentation</title>
<link rel="canonical" href="$stable_url$page">
<meta http-equiv="refresh" content="0; url=$stable_url$page">
</head>
<body><p>This page is in the <a href="$stable_url$page">stable documentation</a>.</p></body>
</html>
EOF
        echo "redirect: /$page -> $stable_url$page"
    done < <(cd "$site/$stable_path" && find . -name '*.html' -not -path './_*' | sort)
fi
# Every version's switcher reads the list served at the root; serve the
# one this deploy was planned from (main's), whatever the root build held.
mkdir -p "$site/_static"
cp _static/switcher.json "$site/_static/switcher.json"
cp CNAME "$site/CNAME"
