#!/usr/bin/env bash
# Advance the submodule external/kravatskiy (A. Kravatskiy's repository) to its upstream head and
# say what moved: old and new commit, and whether RESULTS.md changed (with the changed rows).
# A plain `git pull` in this repository does NOT do this. Pinned verifications are unaffected:
# they read detached checkouts made by tools/kravatskiy/pinned.py.
#   scripts/update_kravatskiy.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
[ -e external/kravatskiy/.git ] || git submodule update --init external/kravatskiy >/dev/null
OLD="$(git -C external/kravatskiy rev-parse HEAD)"
git submodule update --remote external/kravatskiy
NEW="$(git -C external/kravatskiy rev-parse HEAD)"
echo "external/kravatskiy: ${OLD} -> ${NEW}"
if [ "${OLD}" = "${NEW}" ]; then
  echo "no new commits; RESULTS.md unchanged"
elif git -C external/kravatskiy diff --quiet "${OLD}" "${NEW}" -- RESULTS.md; then
  echo "$(git -C external/kravatskiy rev-list --count "${OLD}..${NEW}") new commit(s); RESULTS.md unchanged"
else
  echo "$(git -C external/kravatskiy rev-list --count "${OLD}..${NEW}") new commit(s); RESULTS.md CHANGED. Changed rows (- old, + new):"
  git -C external/kravatskiy diff -U0 "${OLD}" "${NEW}" -- RESULTS.md | grep -E '^[-+]\|' | cut -c1-160
fi
echo "to record the new submodule commit in this repository: git add external/kravatskiy && git commit"
