#!/usr/bin/env bash
# lint_gate.sh — the curated lint/type gate (WPCLEAN Phase LG, executing CARD-LINT-GATE / R98).
#
# EVERY rule this gate enforces is (i) ZERO at the adoption commit (R98's no-lying-gate law:
# no gate over a known-dirty baseline) and (ii) tied to a NAMED defect class from THIS repo's
# history — never the advisory backlog wholesale. The mapping:
#
#   rule / check            named defect class it would have caught here
#   ---------------------   ------------------------------------------------------------------
#   F (incl. F601)          the ca237d2 incident: a 20-key registry block duplicated verbatim
#                           in test_every_key_has_consumer.py — the dict literal silently
#                           collapsed, tests stayed green, ruff carried 20 F601 findings for
#                           four commits while nothing read them (the incident that ratified
#                           this card, R98).
#   invalid-syntax @ py311  3.12-isms under the 3.11 floor: preflight_mint.py:952 could not
#   (ruff parser +          PARSE on the pinned CI interpreter (gate 12 dead on 3.11) and
#   pyright py 3.11)        test_armed_abort_manifest.py used 3.12-only tokenize attrs — both
#                           live at the WPCLEAN census, both invisible on the 3.13 dev venv.
#   PLE                     PLE0303: CorpusSource.__len__ -> int | None, a live TypeError-in-
#                           waiting at census (fixed same-phase).
#   E/W/B/BLE/I/UP          zeroed by the Phase LT burn-down and held at zero here so the
#                           select list in pyproject stays an enforced claim, not advisory
#                           fog. BLE is repo_design §11's own ban. E501 is dispositioned
#                           NEVER and ignored in config (CENSUS_LT §7) — not enforced here.
#   pyright (basic,         the None-flow / wrong-shape class: 57 basic-mode src findings at
#   src+tools, ZERO)        census including reachable TypeErrors (see IMPL_NOTES_LT_PYRIGHT).
#
# The pyright UNDECLARED-MEMBER class (called-and-undeclared on a protocol seam — TD-1's
# class) is deliberately NOT adopted here: the AST conformance gate
# (tests/train/test_trainer_seam_conformance.py) covers it natively, seam-scoped and
# mutation-tested (R106). Strict-mode pyright is CARDED adopt-later (CARD-PYRIGHT-STRICT),
# not enforced: 71.6% of its output was measured config-artifact noise (CENSUS_LT §5b).
#
# Self-test (LAW-07: the gate must be able to fire): --self-test plants one violation per
# arm through stdin/scratch fixtures and requires each arm to go RED, then re-runs the real
# gate. A gate whose trigger cannot fire is a phantom input (LAW-07's own class).
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

run_ruff() { uv run ruff check . ; }
run_pyright_count() {
  uv run pyright --outputjson 2>/dev/null \
    | uv run python -c 'import json,sys; print(json.load(sys.stdin)["summary"]["errorCount"])'
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
  echo "lint_gate self-test: all 3 arms fire"
  return 0
}

if [ "${1:-}" = "--self-test" ]; then
  self_test || exit 1
fi

echo "lint_gate: ruff (curated select, pyproject-authoritative)"
run_ruff || { echo "lint_gate: RUFF RED" >&2; exit 1; }

echo "lint_gate: pyright (basic, src+tools, zero-error baseline)"
ERRS="$(run_pyright_count)"
if [ "${ERRS:-1}" != "0" ]; then
  echo "lint_gate: PYRIGHT RED (${ERRS:-unreadable} errors; baseline is 0)" >&2
  exit 1
fi

echo "lint_gate: GREEN"
