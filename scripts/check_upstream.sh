#!/bin/sh
# Warn (never fail) when a Ginkgo checkout does not match the ref this docs
# branch documents. Building from the wrong ref silently produces an API
# reference and example listings for a different Ginkgo version.
# Usage: check_upstream.sh <ginkgo-source-dir> <pinned-ref>
src=$1
ref=$2
if [ -z "$ref" ]; then
    echo "warning: no .ginkgo-ref in this docs branch; cannot verify $src" >&2
    exit 0
fi
if ! git -C "$src" rev-parse --git-dir >/dev/null 2>&1; then
    echo "note: $src is not a git checkout; cannot verify it is at $ref" >&2
    exit 0
fi
tag=$(git -C "$src" describe --tags --exact-match HEAD 2>/dev/null)
branch=$(git -C "$src" rev-parse --abbrev-ref HEAD)
if [ "$tag" != "$ref" ] && [ "$branch" != "$ref" ]; then
    actual=$(git -C "$src" describe --tags --always 2>/dev/null)
    echo "warning: $src is at $branch ($actual), but this docs branch documents $ref" >&2
    echo "         to match: git -C $src checkout $ref" >&2
fi
if [ -n "$(git --no-optional-locks -C "$src" status --porcelain --untracked-files=no)" ]; then
    echo "warning: $src has uncommitted changes; the generated docs include them" >&2
fi
exit 0
