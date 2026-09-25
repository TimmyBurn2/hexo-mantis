#!/usr/bin/env bash
# CI gate 3c: COUNT collected >= floor(ref) and MONOTONICITY floor(tree) >= floor(ref), both every run;
# the count alone is satisfiable by lowering the floor. A count from an interrupted collection is refused.
# Ref order: origin/dev, local dev, a shallow fetch, then a LOUD bootstrap arm that enforces nothing.
# `--collected N` / `--pytest-cmd CMD` let tests/tools/test_test_count_gate.py drive THIS script; each
# announces itself, and no CI step passes either.
set -euo pipefail
# THE GATES NEVER RE-SYNC THE VENV: `uv run` inherits uv's own `--no-sync`.
export UV_NO_SYNC=1

FLOOR_FILE="tools/ci_gates/test_count_floor.txt"
# The ONE sanctioned way the floor goes DOWN: one `<from> -> <to> <grounds>` record, valid only when both
# numbers match the floors in play and the tree collects exactly `<to>`; a record left behind reds BY NAME.
RATCHET_FILE="tools/ci_gates/test_count_ratchet_down.txt"
MAIN_BRANCH="dev"
PYTEST_CMD="uv run pytest"
# The interpreter the tier census runs under. `uv run` so it sees the project env, the
# same way the collection above does; overridable for the harness.
PYTHON_BIN=${PYTHON_BIN:-"uv run python"}
#: Lines of the collection log echoed on a refusal. Enough to carry pytest's
#: `short test summary info` block, which is where the failing module is named.
LOG_TAIL=40

die() { printf 'gate 3c: %s\n' "$*" >&2; exit 2; }
is_uint() { [[ $1 =~ ^[0-9]+$ ]]; }

# The decision, isolated from every source of input so the self-test can drive it directly.
# args: collected ref_floor tree_floor ref_label ; rc 0 = clean, 1 = violation.
verdict() {
  local count=$1 ref_floor=$2 tree_floor=$3 ref=$4 rc=0
  local sanctioned=${RATCHET_RECORD:-}
  if [ -n "$sanctioned" ]; then
    local r_from r_to
    r_from=${sanctioned%%->*}; r_from=${r_from//[[:space:]]/}
    r_to=${sanctioned#*->}; r_to=${r_to%%[!0-9 ]*}; r_to=${r_to//[[:space:]]/}
    if ! is_uint "$r_from" || ! is_uint "$r_to"; then
      printf 'gate 3c FAIL (ratchet-down): %s is not a `<from> -> <to> <grounds>` record: %s\n' \
        "$RATCHET_FILE" "$sanctioned"
      return 1
    fi
    if [ "$r_from" != "$ref_floor" ] || [ "$r_to" != "$tree_floor" ] \
       || [ "$count" != "$tree_floor" ] || [ "$tree_floor" -ge "$ref_floor" ]; then
      printf 'gate 3c FAIL (ratchet-down): STALE record %s -> %s in %s.\n' \
        "$r_from" "$r_to" "$RATCHET_FILE"
      printf '  This run has ref_floor=%s tree_floor=%s collected=%s. A ratchet-down record\n' \
        "$ref_floor" "$tree_floor" "$count"
      printf '  authorises exactly ONE decrease, states both of its numbers, and is deleted in\n'
      printf '  the commit after it lands. Delete it, or make it name this decrease.\n'
      return 1
    fi
    printf 'gate 3c: SANCTIONED RATCHET-DOWN %s -> %s (%s). The non-decreasing property is\n' \
      "$r_from" "$r_to" "${sanctioned#*->}"
    printf '  DELIBERATELY waived for this one decrease and is enforced again on the next run.\n'
    if [ "$tree_floor" -gt "$count" ]; then
      printf 'gate 3c FAIL (over-ratchet): %s is %s but only %s test(s) collected.\n' \
        "$FLOOR_FILE" "$tree_floor" "$count"
      rc=1
    fi
    return "$rc"
  fi
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
  # The tree's own floor above its collection claims tests that do not exist, and the two arms
  # above stay green while the ref floor trails it.
  if [ "$tree_floor" -gt "$count" ]; then
    printf 'gate 3c FAIL (over-ratchet): %s is %s in this tree but only %s test(s) collected.\n' \
      "$FLOOR_FILE" "$tree_floor" "$count"
    printf '  A floor above the collection is a floor over tests that are not there. Ratchet\n'
    printf '  it to what this tree collects, or find the suite that went missing.\n'
    rc=1
  fi
  return "$rc"
}

# The COLLECTION decision, isolated the same way. args: pytest_rc summary_line ; rc 0 = trusted.
# Two arms on purpose: the exit status, and the summary text for when a pipeline swallows it.
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

# The trigger proves it can fire, on every invocation, before its verdict is trusted; arm 5 is
# a real regression laundered by editing the floor down to meet it.
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
  # The tree's own floor ratcheted PAST what the tree collects. Both other arms stay quiet —
  # the count clears the REF floor and the tree floor only went up.
  _expect_fail  "7b floor above the collection" 95 90 100 "FAIL (over-ratchet)"
  _expect_clean "7c floor equal to the collection" 95 90 95

  # Arms 7d-7g drive the SANCTIONED RATCHET-DOWN. Each is a shape the record must or must
  # not authorise, and 7e/7f are the two ways a record could otherwise launder a regression.
  RATCHET_RECORD="90 -> 80  R346(f) self-test" _expect_clean "7d sanctioned decrease" 80 90 80
  RATCHET_RECORD="90 -> 80  R346(f) self-test" _expect_fail \
    "7e record with slack (a real regression under a true record)" 75 90 80 "FAIL (ratchet-down)"
  RATCHET_RECORD="95 -> 80  R346(f) self-test" _expect_fail \
    "7f record naming another decrease" 80 90 80 "STALE record"
  RATCHET_RECORD="90 -> 91  R346(f) self-test" _expect_fail \
    "7g record claiming an increase" 91 90 91 "STALE record"
  RATCHET_RECORD="not a record" _expect_fail \
    "7h malformed record" 80 90 80 "is not a"

  # Arms 8-11 drive `collection_verdict`. Arm 9 is VERBATIM the summary line and exit status
  # this tree printed under a planted import break.
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

# Echoes a git revision usable as `git show <rev>:<path>`, or nothing. A stale local dev never
# outranks a fetched origin/dev; the fetch is last as the only arm with a side effect.
resolve_ref() {
  if git rev-parse --verify -q "refs/remotes/origin/$MAIN_BRANCH^{commit}" >/dev/null; then
    printf 'origin/%s' "$MAIN_BRANCH"; return 0
  fi
  if git rev-parse --verify -q "refs/heads/$MAIN_BRANCH^{commit}" >/dev/null; then
    printf '%s' "$MAIN_BRANCH"; return 0
  fi
  # Arm 3: a depth-1 CI checkout has neither ref (`fetch-depth: 0` in ci.yml would retire this).
  # Best effort and never fatal: an offline clone falls through to the bootstrap arm.
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
    # The arms count themselves: a transcribed tally goes wrong the first time an arm is added.
    local clean fired
    # `|| true`: `grep -c` exits 1 on a zero count and `set -e` would kill the run; a zero
    # prints as a zero and is visibly wrong.
    clean=$(grep -cE '^ *_(expect|collection)_clean +"' "$0" || true)
    fired=$(grep -cE '^ *_(expect|collection)_fail +"' "$0" || true)
    echo "gate 3c self-test: $clean clean arms + $fired firing arms, all correct"
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

  RATCHET_RECORD=""
  if [ -f "$RATCHET_FILE" ]; then
    RATCHET_RECORD=$(grep -v '^[[:space:]]*#' "$RATCHET_FILE" | grep -v '^[[:space:]]*$' | head -1)
    [ -n "$RATCHET_RECORD" ] || die "$RATCHET_FILE exists but holds no record"
  fi
  export RATCHET_RECORD

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
    # BOTH streams into the log and the status kept: a count from a collection that died is not one.
    local log pytest_rc=0 summary
    log=$(mktemp) || die "could not create a temp file for the collection log"
    # shellcheck disable=SC2086  # PYTEST_CMD is a command line, deliberately word-split.
    # `-m ''` clears addopts' default-tier marker: the gate counts the WHOLE tree, not the tier.
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
  # Both arms run even when the first reds, so one run reports both facts.
  local count_rc=0 census_rc=0
  verdict "$collected" "$ref_floor" "$tree_floor" "$ref" || count_rc=$?
  # The tier census rides here: a deselecting marker loses a test from every tier yet leaves it counted.
  # shellcheck disable=SC2086  # PYTHON_BIN is a command line, deliberately word-split.
  $PYTHON_BIN "$(dirname "$0")/tier_census.py" || census_rc=$?
  if [ "$count_rc" -ne 0 ] || [ "$census_rc" -ne 0 ]; then
    return 1
  fi
  return 0
}

main "$@"
