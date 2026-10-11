#!/usr/bin/env bash
# Print the deploy build matrix for a switcher.json: one
# {version, branch, url, path} object per entry, where `path` is the entry's
# URL relative to the site root ("" for the root itself).
# Usage: versions_matrix.sh _static/switcher.json
set -euo pipefail
jq -c '[.[] | {version, branch, url,
              path: (.url | ltrimstr("https://gko-project.org/") | rtrimstr("/"))}]' "$1"
