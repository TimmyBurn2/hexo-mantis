#!/usr/bin/env bash
# CI gate 3c: the collected-test count is non-decreasing, AND so is the floor itself.
#
# WHY THIS FILE WAS REWRITTEN (the defect, measured on branch `remediation`).
# The predecessor resolved its comparison ref as `origin/main`, then `main`, then fell back
# to `cat` of the floor file in the WORKING TREE. This repo has no `main` branch and never
# had one -- `git branch -a` lists `dev`, `remediation`, `wppre-scratch`, and CLAUDE.md names
# `dev` as the main branch ("Main branch (you will usually use this for PRs): dev").
# `git rev-parse --verify -q origin/main` and `... main` therefore both exited 1 on EVERY
# invocation since WP0, and the gate compared the collected count against the floor file
# sitting next to it in the same tree. Two consequences:
#   * the advertised "non-decreasing vs main" property never existed at all; and
#   * a commit that LOWERED the floor passed trivially -- editing the floor down was, in
#     practice, the way to turn this gate from red to green.
#
# THE TWO PROPERTIES NOW ENFORCED, both against a REAL ref:
#   1. COUNT       collected >= floor(ref)   -- the original, now with a ref that resolves.
#   2. MONOTONICITY floor(tree) >= floor(ref) -- the half that did not exist. Property 1
#      alone is satisfiable by lowering the floor, so property 1 alone enforces nothing.
# Each prints its own distinct, actionable message; both are checked on every run, so a
# commit that trips both is told about both rather than one at a time.
#
# PRECEDENCE ORDER FOR THE REF, AND WHY IT IS THIS ORDER:
#   1. `origin/dev` when the ref already exists locally. It is the branch a PR is merged
#      into, so it is the thing "non-decreasing" is supposed to mean, and consulting an
#      already-fetched ref costs no network.
#   2. local `dev`. Used only when `origin/dev` is absent, so a stale local `dev` can never
#      outrank a fetched remote one. This is the arm a clone with no remote uses, and this
#      repo spent most of its life with no git remote at all.
#   3. a shallow `git fetch origin dev` -> FETCH_HEAD. LAST, because it is the only arm with
#      a side effect and the only one that can hang. It exists because `actions/checkout@v4`
#      defaults to `fetch-depth: 1` + single-branch, which leaves a CI job with NEITHER
#      `origin/dev` NOR local `dev` -- i.e. without this arm the CI run, the one run that
#      matters, would take the bootstrap arm and enforce nothing. See the note at the foot
#      of this comment.
#   4. bootstrap: no ref exists. A genuine fresh clone of a repo whose history does not yet
#      contain the main branch. It compares against the working-tree floor, monotonicity is
#      VACUOUS (there is nothing to be monotone against), and it says so LOUDLY on both
#      stdout and stderr, with `ref=<none:bootstrap>` in the summary line. The predecessor's
#      failure was not that it had a bootstrap arm; it was that the bootstrap arm was silent
#      and therefore indistinguishable from a real comparison in a CI log.
#
# NOTE for whoever owns `.github/workflows/ci.yml`: arm 3 is a repair, not a design. The
# right fix is `fetch-depth: 0` (or an explicit `git fetch origin dev`) on the python job's
# checkout step, after which arm 1 fires and this script never touches the network.
#
# THE SECOND DEFECT, F-816-33, fixed 2026-09-02 and measured on this tree before the fix.
# The measuring line read
#     collected=$(uv run pytest --collect-only -q 2>/dev/null | grep -Eo '...collected' ...) || true
# and pytest, on a module that fails to IMPORT, prints `4414 tests collected, 1 error in 2.08s`
# and exits 2. BOTH signals were discarded -- the status by the pipeline and the `|| true`, the
# error text by `2>/dev/null` -- and the grep matched the count inside the very line saying the
# collection was interrupted. The gate printed `collected=4414 ... GATE_RC=0`: a PASS on a tree
# whose collection had died. The one case it caught was collection so broken it printed no
# number at all, i.e. the loud one; the quiet one is the one that loses tests.
# NOW: both streams go to a LOG, the exit status is captured, and `collection_verdict` refuses
# a count taken from an interrupted collection. A count is only a count of the tree if the
# collection that produced it finished.
#
# `--collected N` injects the count instead of measuring it, and `--pytest-cmd CMD` replaces
# the collection command. Both exist so the producer test (tests/tools/test_test_count_gate.py,
# LAW-07) can drive THIS script -- not a re-implemented copy of its decision -- inside throwaway
# git repos: the first for the floor comparison, the second so the broken-collection arm can be
# driven without breaking the real suite. Each announces itself on stdout so an injected value
# can never be mistaken for a measured one in a log, and no CI step passes either (pinned by
# tests/tools/test_test_count_gate.py's parse of every `run:` body).
set -euo pipefail

FLOOR_FILE="tools/ci_gates/test_count_floor.txt"
MAIN_BRANCH="dev"
PYTEST_CMD="uv run pytest"
#: Lines of the collection log echoed on a refusal. Enough to carry pytest's
#: `short test summary info` block, which is where the failing module is named.
LOG_TAIL=40

die() { printf 'gate 3c: %s\n' "$*" >&2; exit 2; }
is_uint() { [[ $1 =~ ^[0-9]+$ ]]; }

# ---------------------------------------------------------------------------------------
# The decision, isolated from every source of input so the self-test can drive it directly.
# args: collected ref_floor tree_floor ref_label ; rc 0 = clean, 1 = violation.
# ---------------------------------------------------------------------------------------
verdict() {
  local count=$1 ref_floor=$2 tree_floor=$3 ref=$4 rc=0
  if [ "$count" -lt "$ref_floor" ]; then
    printf 'gate 3c FAIL (count): collected %s test(s), floor is %s at %s.\n' \
      "$count" "$ref_floor" "$ref"
    printf '  Tests were LOST. Find the deleted/skipped suite. Lowering %s is NOT the fix --\n' \
      "$FLOOR_FILE"
    printf '  the monotonicity check below exists because that was the habit.\n'
    rc=1
  fi
  if [ "$tree_floor" -lt "$ref_floor" ]; then
    printf 'gate 3c FAIL (monotonicity): %s is %s in this tree but %s at %s.\n' \
      "$FLOOR_FILE" "$tree_floor" "$ref_floor" "$ref"
    printf '  The floor may only ratchet UP. A lowered floor makes a red gate green while\n'
    printf '  deleting the evidence that anything was lost; that is the defect this arm closes.\n'
    rc=1
  fi
  return "$rc"
}

# ---------------------------------------------------------------------------------------
# The COLLECTION decision, isolated from the collection the same way.
# args: pytest_rc summary_line ; rc 0 = the count may be trusted, 1 = it may not.
# Two independent arms on purpose: the exit status is the primary signal, and the summary
# text catches the case where something between pytest and this script swallows the status
# -- which is exactly what the predecessor's pipeline did.
# ---------------------------------------------------------------------------------------
collection_verdict() {
  local rc=$1 summary=$2 bad=0
  if [ "$rc" -ne 0 ]; then
    printf 'gate 3c FAIL (collection): pytest --collect-only exited %s.\n' "$rc"
    bad=1
  fi
  if printf '%s\n' "$summary" | grep -Eqi '[0-9]+ error'; then
    printf 'gate 3c FAIL (collection): the summary line reports collection errors: %s\n' \
      "$summary"
    bad=1
  fi
  if [ "$bad" -ne 0 ]; then
    printf '  A count from an INTERRUPTED collection is not a count of the tree -- it is the\n'
    printf '  number of tests that survived whatever broke. A whole file can vanish from the\n'
    printf '  count this way and leave the number looking clean. Fix the collection first.\n'
    return 1
  fi
  return 0
}

# ---------------------------------------------------------------------------------------
# LAW-07: the trigger proves it can fire, on every invocation, before its verdict is trusted.
# Each arm is a shape the gate exists to catch, and arm 5 is the exact shape the predecessor
# passed: a real regression laundered by editing the floor down to meet it.
# ---------------------------------------------------------------------------------------
self_test() {
  local failures=0 out

  _expect_clean() {  # label count ref tree
    local label=$1
    if ! out=$(verdict "$2" "$3" "$4" self-test 2>&1); then
      printf '    arm %s: fired on a legitimate tree -- %s\n' "$label" "$out" >&2
      failures=$((failures + 1))
    fi
  }
  _expect_fail() {   # label count ref tree needle
    local label=$1 needle=$5
    if out=$(verdict "$2" "$3" "$4" self-test 2>&1); then
      printf '    arm %s: did NOT fire (count=%s ref=%s tree=%s)\n' "$label" "$2" "$3" "$4" >&2
      failures=$((failures + 1))
    elif [[ $out != *"$needle"* ]]; then
      printf '    arm %s: fired without the %s message -- %s\n' "$label" "$needle" "$out" >&2
      failures=$((failures + 1))
    fi
  }

  _expect_clean "1 count above floor"       100 90 90
  _expect_clean "2 count equals floor"       90 90 90
  _expect_clean "3 floor ratcheted up"      100 90 95
  _expect_fail  "4 count below floor"        80 90 90 "FAIL (count)"
  _expect_fail  "5 floor lowered"           100 90 89 "FAIL (monotonicity)"
  # The predecessor's exact hole: tests were lost AND the floor was edited down to match.
  # Against the working-tree floor that reads as green; against a real ref it is two faults.
  _expect_fail  "6 lost tests + lowered floor" 80 90 80 "FAIL (count)"
  _expect_fail  "7 lost tests + lowered floor" 80 90 80 "FAIL (monotonicity)"

  # Arms 8-11 drive `collection_verdict`. Arm 9 is F-816-33 VERBATIM: the summary line this
  # tree actually printed under a planted import break, beside the status pytest actually
  # exited with.
  _collection_clean() {  # label rc summary
    if ! out=$(collection_verdict "$2" "$3" 2>&1); then
      printf '    arm %s: fired on a finished collection -- %s\n' "$1" "$out" >&2
      failures=$((failures + 1))
    fi
  }
  _collection_fail() {   # label rc summary
    if out=$(collection_verdict "$2" "$3" 2>&1); then
      printf '    arm %s: did NOT fire (rc=%s summary=%s)\n' "$1" "$2" "$3" >&2
      failures=$((failures + 1))
    fi
  }
  _collection_clean "8 finished collection"   0 "4414 tests collected in 2.08s"
  _collection_fail  "9 F-816-33 shape"        2 "4414 tests collected, 1 error in 2.08s"
  _collection_fail  "10 status swallowed"     0 "4414 tests collected, 1 error in 2.08s"
  _collection_fail  "11 no tests collected"   5 "no tests ran in 0.51s"

  unset -f _expect_clean _expect_fail _collection_clean _collection_fail
  if [ "$failures" -ne 0 ]; then
    printf 'gate 3c SELF-TEST FAIL -- the trigger cannot be trusted (%s arm(s)):\n' \
      "$failures" >&2
    return 1
  fi
  return 0
}

# ---------------------------------------------------------------------------------------
# Ref resolution. Echoes a git revision usable as `git show <rev>:<path>`, or nothing.
# ---------------------------------------------------------------------------------------
resolve_ref() {
  if git rev-parse --verify -q "refs/remotes/origin/$MAIN_BRANCH^{commit}" >/dev/null; then
    printf 'origin/%s' "$MAIN_BRANCH"; return 0
  fi
  if git rev-parse --verify -q "refs/heads/$MAIN_BRANCH^{commit}" >/dev/null; then
    printf '%s' "$MAIN_BRANCH"; return 0
  fi
  # Arm 3 -- the shallow-checkout repair. Best effort and never fatal: an offline clone
  # must fall through to the bootstrap arm rather than die here.
  if git remote get-url origin >/dev/null 2>&1 \
     && git fetch --quiet --depth=1 origin "$MAIN_BRANCH" >/dev/null 2>&1 \
     && git rev-parse --verify -q 'FETCH_HEAD^{commit}' >/dev/null; then
    printf 'FETCH_HEAD'; return 0
  fi
  return 1
}

main() {
  local collected="" self_test_only=0 pytest_cmd_injected=0
  while [ $# -gt 0 ]; do
    case $1 in
      --collected) collected=${2:-}; shift 2 ;;
      --pytest-cmd) PYTEST_CMD=${2:-}; pytest_cmd_injected=1; shift 2 ;;
      --self-test) self_test_only=1; shift ;;
      *) die "unknown argument: $1 (usage: $0 [--collected N] [--pytest-cmd CMD] \
[--self-test])" ;;
    esac
  done

  self_test || exit 1
  if [ "$self_test_only" -eq 1 ]; then
    echo "gate 3c self-test: 4 clean arms + 7 firing arms, all correct"
    return 0
  fi

  local toplevel
  toplevel=$(git rev-parse --show-toplevel 2>/dev/null) \
    || die "not inside a git work tree; gate 3c compares against a git ref"
  cd "$toplevel"
  [ -f "$FLOOR_FILE" ] || die "$FLOOR_FILE is missing from the working tree"

  local tree_floor
  tree_floor=$(tr -d '[:space:]' < "$FLOOR_FILE")
  is_uint "$tree_floor" || die "$FLOOR_FILE in this tree is not a count: '$tree_floor'"

  local ref ref_floor
  ref=$(resolve_ref) || ref=""
  if [ -n "$ref" ]; then
    if ! ref_floor=$(git show "$ref:$FLOOR_FILE" 2>/dev/null | tr -d '[:space:]'); then
      printf 'gate 3c WARNING: %s resolves but carries no %s -- it predates gate 3c.\n' \
        "$ref" "$FLOOR_FILE" >&2
      ref=""
    elif ! is_uint "$ref_floor"; then
      die "$FLOOR_FILE at $ref is not a count: '$ref_floor'"
    fi
  fi

  if [ -z "$ref" ]; then
    # Arm 4. LOUD by construction: silence here is what made the predecessor useless.
    ref="<none:bootstrap>"
    ref_floor=$tree_floor
    printf 'gate 3c WARNING: no comparison ref (no origin/%s, no local %s, no fetch).\n' \
      "$MAIN_BRANCH" "$MAIN_BRANCH" >&2
    printf '  Comparing against the working-tree floor. The non-decreasing property is NOT\n' >&2
    printf '  enforced this run and the monotonicity check is vacuous. Fetch %s to restore it.\n' \
      "$MAIN_BRANCH" >&2
    echo "gate 3c: BOOTSTRAP ARM -- no ref, monotonicity NOT enforced this run"
  fi

  if [ "$pytest_cmd_injected" -eq 1 ]; then
    echo "gate 3c: collection command INJECTED via --pytest-cmd (test-harness path)"
  fi
  if [ -n "$collected" ]; then
    is_uint "$collected" || die "--collected wants a non-negative integer, got '$collected'"
    echo "gate 3c: collected count INJECTED via --collected (test-harness path, not measured)"
  else
    local log pytest_rc=0 summary
    log=$(mktemp) || die "could not create a temp file for the collection log"
    # BOTH streams into the log, and the status kept. The predecessor sent stderr to
    # /dev/null and lost the status to a pipeline; that is the whole of F-816-33.
    # shellcheck disable=SC2086  # PYTEST_CMD is a command line, deliberately word-split.
    # `-m ''` CLEARS the default-tier marker expression pyproject's addopts now carries (R330(g)):
    # this gate counts the WHOLE tree, and deselection is not collection. Without it the count
    # would still parse right by regex accident (`N/M tests collected` matches on M) and be one
    # summary-format change away from reading the filtered N.
    $PYTEST_CMD --collect-only -q -m '' >"$log" 2>&1 || pytest_rc=$?
    summary=$(grep -Ei '[0-9]+ (tests? collected|errors?)|no tests ran' "$log" | tail -1)
    if ! collection_verdict "$pytest_rc" "$summary"; then
      sed -e 's/^/  | /' "$log" | tail -"$LOG_TAIL" >&2
      rm -f "$log"
      exit 1
    fi
    collected=$(grep -Eo '[0-9]+ tests? collected' "$log" | grep -Eo '^[0-9]+' | tail -1) || true
    rm -f "$log"
    is_uint "${collected:-}" \
      || die "could not read a collected-test count from pytest (got '${collected:-}'); \
collection itself is probably broken, which is a worse failure than this gate"
  fi

  echo "collected=$collected floor=$ref_floor ref=$ref tree_floor=$tree_floor"
  verdict "$collected" "$ref_floor" "$tree_floor" "$ref"
}

main "$@"
