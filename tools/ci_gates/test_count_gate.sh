#!/usr/bin/env bash
# CI gate 3c: collected-test count non-decreasing. Until a `main` branch exists, the
# floor is the committed tools/ci_gates/test_count_floor.txt (seeded by WP0 with the
# measured count); once main exists, the floor is that file AS OF main, so every PR is
# non-decreasing vs main and each work package bumps the floor file with its own count.
set -euo pipefail
floor_file="tools/ci_gates/test_count_floor.txt"
if git rev-parse --verify -q origin/main >/dev/null; then
  floor=$(git show origin/main:"$floor_file")
elif git rev-parse --verify -q main >/dev/null; then
  floor=$(git show main:"$floor_file")
else
  floor=$(cat "$floor_file")   # pre-main bootstrap: committed floor seeded by WP0
fi
count=$(uv run pytest --collect-only -q 2>/dev/null | grep -Eo '[0-9]+ tests? collected' | grep -Eo '^[0-9]+' | tail -1)
echo "collected=$count floor=$floor"
test "$count" -ge "$floor"
