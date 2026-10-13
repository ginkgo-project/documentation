#!/usr/bin/env bash
# Print the deploy build matrix for a switcher.json: one
# {version, branch, url, path, stable_url} object per entry, where `path` is
# the entry's URL relative to the site root ("" for the root itself) and
# `stable_url` is the URL of the preferred entry (the stable docs).
# Usage: versions_matrix.sh _static/switcher.json
set -euo pipefail
jq -c '(map(select(.preferred)) | first | .url) as $stable
       | [.[] | {version, branch, url,
                 path: (.url | ltrimstr("https://gko-project.org/") | rtrimstr("/")),
                 stable_url: $stable}]' "$1"
