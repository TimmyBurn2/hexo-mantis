#!/usr/bin/env bash
# The curated lint/type gate (CARD-LINT-GATE). rc 0 clean and both tools RAN; rc 1 a rule is red;
# rc 2 REFUSAL, a tool could not run or gave no summary: a missing interpreter is not a type error.
# Each rule is zero at adoption and names the defect class it would have caught here:
#   F (F601) a duplicated dict key silently collapsed a test registry; py311 syntax could not parse
#   on the pinned 3.11 interpreter; PLE0303 a `__len__ -> int | None`; E/W/B/BLE/I/UP held at zero;
#   pyright basic the None-flow / wrong-shape class. Seam members and strict mode are not adopted:
#   tests/train/test_trainer_seam_conformance.py covers the first; the second is carded (config noise).
set -u
# THE GATES NEVER RE-SYNC THE VENV: `uv run` inherits uv's own `--no-sync`.
export UV_NO_SYNC=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

run_ruff() { uv run ruff check . ; }
run_pyright_count() {
  # pipefail, or a pyright that dies mid-JSON reports the reader's status and reads green. Subshell-
  # scoped: this file's `printf ... | grep -q` idioms want the DOWNSTREAM status.
  ( set -o pipefail
    uv run pyright --outputjson 2>/dev/null \
      | uv run python -c 'import json,sys; s=json.load(sys.stdin)["summary"]; print(s["errorCount"], s.get("filesAnalyzed", -1))' )
}

# `errorCount: 0` over zero analysed files is a green over nothing. The floor catches "nothing",
# not the file count, so it needs no re-editing as the tree grows.
PYRIGHT_MIN_FILES=100

# A missing node behind the pyright shim yields no JSON, like a broken config, so it REFUSES with a
# named cause. Pure over probe text, so the self-test drives it on a healthy host.
PYRIGHT_REFUSED_RC=2

_classify_pyright_probe() {
  local out="$1" rc="${2:-0}"
  if printf '%s' "$out" | grep -qiE 'pyright[[:space:]]+[0-9]+\.[0-9]+'; then
    return 0
  fi
  echo "lint_gate: REFUSING -- pyright is not runnable, so this gate cannot report on types." >&2
  echo "  named cause: the pyright shim produced no version string (probe rc ${rc})." >&2
  if printf '%s' "$out" | grep -qiE 'no version is set for shim|command not found|ENOENT|not found'; then
    echo "  most likely: the node interpreter behind the shim has no selected version." >&2
    echo "  fix: select one (e.g. a mise global node version), or put an installed node bin" >&2
    echo "       on PATH for this invocation. This is a HOST condition, not a repo defect." >&2
  fi
  echo "  probe output: ${out}" >&2
  echo "  This is a REFUSAL (rc ${PYRIGHT_REFUSED_RC}), NOT a lint red (rc 1): nothing was" >&2
  echo "  measured, so nothing may be reported -- green or red (R289(u))." >&2
  return 1
}

pyright_preflight() {
  local out rc
  out="$(uv run pyright --version 2>&1)"; rc=$?
  _classify_pyright_probe "$out" "$rc"
}

self_test() {
  # Arm 1 — F601 (the seed incident's rule) must red on a duplicated literal key.
  if printf 'x = {1: "a", 1: "b"}\n' \
      | uv run ruff check --stdin-filename src/mantis/_lint_gate_selftest.py --select F - >/dev/null 2>&1; then
    echo "lint_gate SELF-TEST FAIL: F601 fixture did not red" >&2; return 1
  fi
  # Arm 2 — the 3.11 floor: 3.12-only syntax must red under target-version py311.
  if printf 'x = f"{1 if True\n else 2}"\n' \
      | uv run ruff check --stdin-filename src/mantis/_lint_gate_selftest.py - >/dev/null 2>&1; then
    echo "lint_gate SELF-TEST FAIL: 3.12-only syntax did not red under the py311 floor" >&2; return 1
  fi
  # Arm 4 — a missing interpreter REFUSES with a named cause, not a failed fixture.
  if _classify_pyright_probe "mise ERROR No version is set for shim: node" 1 2>/dev/null; then
    echo "lint_gate SELF-TEST FAIL: missing-interpreter probe did not refuse" >&2; return 1
  fi
  # +control — a real version string must classify as RUNNABLE, or arm 4 would refuse always.
  if ! _classify_pyright_probe "pyright 1.1.400" 0 2>/dev/null; then
    echo "lint_gate SELF-TEST FAIL: a valid version string classified as unrunnable" >&2; return 1
  fi
  # Arm 3 needs a runnable pyright; refuse before blaming its fixture.
  pyright_preflight || exit "$PYRIGHT_REFUSED_RC"
  # Arm 3 — pyright must red on an obvious type error in a scratch project.
  local scratch; scratch="$(mktemp -d)"
  printf '{"include": ["bad.py"], "typeCheckingMode": "basic"}\n' > "$scratch/pyrightconfig.json"
  printf 'def f(x: int) -> int:\n    return x\n\nf("not an int")\n' > "$scratch/bad.py"
  local count
  count="$(uv run pyright -p "$scratch" --outputjson 2>/dev/null \
    | uv run python -c 'import json,sys; print(json.load(sys.stdin)["summary"]["errorCount"])')"
  rm -rf "$scratch"
  if [ "${count:-0}" -eq 0 ]; then
    echo "lint_gate SELF-TEST FAIL: pyright fixture did not red" >&2; return 1
  fi
  echo "lint_gate self-test: all 4 arms fire (+1 control)"
  return 0
}

if [ "${1:-}" = "--self-test" ]; then
  self_test || exit 1
fi

# The comment ratchet runs FIRST because it needs no node: a pyright refusal must not take it down.
echo "lint_gate: comment-length ratchet (R368(g))"
uv run python tools/ci_gates/comment_lint.py \
  || { echo "lint_gate: COMMENT RATCHET RED" >&2; exit 1; }

echo "lint_gate: ruff (curated select, pyproject-authoritative)"
run_ruff || { echo "lint_gate: RUFF RED" >&2; exit 1; }

echo "lint_gate: pyright (basic, src+tools, zero-error baseline)"
pyright_preflight || exit "$PYRIGHT_REFUSED_RC"
PYRIGHT_SUMMARY="$(run_pyright_count)"
ERRS="${PYRIGHT_SUMMARY%% *}"
FILES="${PYRIGHT_SUMMARY##* }"
if [ -z "${ERRS}" ]; then
  echo "lint_gate: REFUSING -- pyright ran but produced no readable errorCount." >&2
  echo "  named cause: --outputjson yielded no parseable summary. Nothing was measured, so" >&2
  echo "  nothing is reported (R289(u)). rc ${PYRIGHT_REFUSED_RC}." >&2
  exit "$PYRIGHT_REFUSED_RC"
fi
# `filesAnalyzed` below the floor is a REFUSAL, not a green: nothing was checked.
if [ "${FILES}" -lt "${PYRIGHT_MIN_FILES}" ] 2>/dev/null; then
  echo "lint_gate: REFUSING -- pyright analysed ${FILES} file(s), below the floor of" >&2
  echo "  ${PYRIGHT_MIN_FILES}. errorCount ${ERRS} over that scope is a green over nothing:" >&2
  echo "  a broken include, an emptied file set, or the wrong working directory. Nothing" >&2
  echo "  was measured, so nothing is reported (R289(u)). rc ${PYRIGHT_REFUSED_RC}." >&2
  exit "$PYRIGHT_REFUSED_RC"
fi
if [ "${ERRS}" != "0" ]; then
  echo "lint_gate: PYRIGHT RED (${ERRS:-unreadable} errors; baseline is 0)" >&2
  exit 1
fi

echo "lint_gate: GREEN (pyright analysed ${FILES} file(s), 0 errors)"
