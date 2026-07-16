#!/usr/bin/env bash
# CI gate 1 (local proof): fresh clone -> uv sync builds the extension -> import works.
set -euo pipefail
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
git clone --quiet . "$tmp/clone"
cd "$tmp/clone"
uv sync --locked
uv run python -c "from mantis import _engine; assert _engine.__doc__; print('gate01 OK:', _engine.hello())"
