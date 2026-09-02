"""CI gate 3c's producer test (LAW-07): the collected-test-count gate must BITE.

>300 justify (R8): ONE gate's producer suite over ONE synthetic-repo fixture. `make_repo` is
the whole file — every case is that one builder parameterised by which comparison ref exists,
and each ref arm's PASS case is the near-miss of another arm's FAIL case (an equal floor passes
where a lowered one must fail; a bootstrap repo passes where a repo with a ref must not). A
reviewer widening any arm has to see the other arms' expectations on the same screen, and
splitting the file would put `make_repo` behind an import that R5 bars from `tests/`. The
self-test and regression-pin sections drive the SAME script through the SAME `run_gate`.

THE DEFECT THIS SUITE EXISTS FOR, measured on branch `remediation`. The gate resolved its
comparison ref as `origin/main`, then `main`, then fell back to `cat` of the floor file in the
WORKING TREE. There is no `main` branch in this repo and there never was (CLAUDE.md: "Main
branch (you will usually use this for PRs): dev"), so both lookups exited 1 on every run since
WP0 and the gate compared the collected count against the floor file sitting next to it. The
"non-decreasing vs main" property was never enforced, and LOWERING the floor -- the one edit
that destroys the evidence a suite was lost -- passed trivially. `test_a_lowered_floor_fails`
below is that exact shape, and it is the arm that did not exist before.

Every case drives the SCRIPT, in a throwaway git repo, through `subprocess`. An oracle that
re-implemented the comparison could drift from the thing it certifies, which is the defect
class this repo keeps finding (gate 11's docstring says so in its own words). `--collected N`
is the injection seam that makes this possible without running pytest inside a synthetic repo;
it announces itself on stdout, and `test_no_ci_step_injects_a_count` pins that no CI step uses
it.

THE SECOND DEFECT, F-816-33, filed 2026-09-01 and fixed 2026-09-02. The rewrite above repaired
the REF and left the COLLECTION unguarded: the count was read from
`uv run pytest --collect-only -q 2>/dev/null | grep ...`, so a module that fails to import made
pytest print `4414 tests collected, 1 error in 2.08s` and exit 2, and the gate discarded the
status (pipeline + `|| true`) and the error text (`2>/dev/null`) and matched the count inside
the very line saying the collection was interrupted. Measured on this tree with a planted
import break BEFORE the fix: `collected=4414 ... GATE_RC=0` -- a PASS over a dead collection,
in the gate whose entire purpose is to notice that tests went missing. `--pytest-cmd` is the
second injection seam, and the `_COLLECT_*` arms below drive the real script's real measuring
path against a fake pytest that reproduces those exact strings.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE = REPO_ROOT / "tools" / "ci_gates" / "test_count_gate.sh"
FLOOR_REL = "tools/ci_gates/test_count_floor.txt"

#: Identity for the synthetic repos. A bare `git commit` inherits the developer's global
#: config, and a machine without one fails the fixture rather than the gate.
_GIT_ID = (
    "-c", "user.name=gate3c-oracle",
    "-c", "user.email=gate3c@example.invalid",
    "-c", "commit.gpgsign=false",
)


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", *_GIT_ID, *args], cwd=repo, capture_output=True, text=True, check=True
    )
    return out.stdout


def _write_floor(repo: Path, value: int) -> None:
    path = repo / FLOOR_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{value}\n", encoding="utf-8")


def _seed(repo: Path, ref_floor: int) -> None:
    """A repo whose `dev` branch carries `ref_floor`, with HEAD moved off `dev`."""
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q", "-b", "dev")
    _write_floor(repo, ref_floor)
    _git(repo, "add", FLOOR_REL)
    _git(repo, "commit", "-q", "-m", "seed floor")
    _git(repo, "checkout", "-q", "-b", "work")


def make_repo(
    tmp_path: Path, ref_floor: int, tree_floor: int, *, ref: str, name: str = "repo"
) -> Path:
    """Build a synthetic repo. `ref` selects which comparison arm is available.

    ``origin``    only `refs/remotes/origin/dev` exists (arm 1)
    ``local``     only `refs/heads/dev` exists (arm 2)
    ``both``      both exist, with DIFFERENT floors, so precedence is observable
    ``fetchable`` an `origin` remote is configured but neither ref exists (arm 3)
    ``none``      no ref at all (arm 4, bootstrap)
    """
    repo = tmp_path / name
    _seed(repo, ref_floor)
    if ref in ("origin", "both", "fetchable"):
        bare = tmp_path / f"{name}-origin.git"
        subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
        _git(repo, "remote", "add", "origin", str(bare))
        _git(repo, "push", "-q", "origin", "dev")
    if ref == "both":
        # A local `dev` that DISAGREES with the remote one, so the arm that wins is visible.
        _git(repo, "branch", "-q", "-f", "dev", "work")
        _write_floor(repo, ref_floor + 1000)
        _git(repo, "add", FLOOR_REL)
        _git(repo, "commit", "-q", "-m", "local dev floor")
        _git(repo, "branch", "-q", "-f", "dev", "HEAD")
        _git(repo, "reset", "-q", "--hard", "HEAD~1")
    if ref in ("origin", "fetchable", "none"):
        _git(repo, "branch", "-q", "-D", "dev")
    if ref in ("fetchable", "none"):
        # `actions/checkout@v4` at its default `fetch-depth: 1` leaves exactly this shape.
        subprocess.run(
            ["git", "update-ref", "-d", "refs/remotes/origin/dev"], cwd=repo, check=False
        )
    _write_floor(repo, tree_floor)
    return repo


def run_gate(repo: Path, collected: int, *, script: Path = GATE) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(script), "--collected", str(collected)],
        cwd=repo, capture_output=True, text=True, check=False,
    )


def _both(proc: subprocess.CompletedProcess) -> str:
    return proc.stdout + proc.stderr


#: Fake collections, as (exit status, stdout). The first two are VERBATIM what this tree
#: printed on 2026-09-02 with and without a planted import break; the third is the same broken
#: summary with the status swallowed, which is the shape the predecessor's pipeline produced.
_COLLECT_CLEAN = (0, "4414 tests collected in 2.08s")
_COLLECT_BROKEN = (2, "ERROR tests/diagnostics/test_planted.py\n"
                      "!!!! Interrupted: 1 error during collection !!!!\n"
                      "4414 tests collected, 1 error in 2.08s")
_COLLECT_STATUS_SWALLOWED = (0, _COLLECT_BROKEN[1])


def run_gate_measuring(
    repo: Path, collection: tuple[int, str], *, script: Path = GATE
) -> subprocess.CompletedProcess:
    """Drive the gate's REAL measuring path with a fake `pytest` standing in for the collection.

    The fake is a shell script rather than a monkeypatch because the gate is a shell script:
    the thing under test is how it treats a command's status and output, and only a real child
    process has either.
    """
    status, text = collection
    fake = repo / "fake_pytest.sh"
    fake.write_text(f"cat <<'EOF'\n{text}\nEOF\nexit {status}\n", encoding="utf-8")
    return subprocess.run(
        ["bash", str(script), "--pytest-cmd", f"bash {fake}"],
        cwd=repo, capture_output=True, text=True, check=False,
    )


# --- the two properties -------------------------------------------------------------------


def test_an_equal_count_and_equal_floor_passes(tmp_path: Path) -> None:
    proc = run_gate(make_repo(tmp_path, 100, 100, ref="origin"), 100)
    assert proc.returncode == 0, _both(proc)
    assert "collected=100 floor=100 ref=origin/dev tree_floor=100" in proc.stdout


def test_a_count_below_the_floor_fails(tmp_path: Path) -> None:
    proc = run_gate(make_repo(tmp_path, 100, 100, ref="origin"), 99)
    assert proc.returncode == 1, _both(proc)
    assert "gate 3c FAIL (count)" in proc.stdout
    assert "gate 3c FAIL (monotonicity)" not in proc.stdout, "the two arms must be distinct"


def test_a_lowered_floor_fails(tmp_path: Path) -> None:
    """THE new property. Under the predecessor this exact repo exited 0.

    Count is comfortably above both floors, so the ONLY fault is that the floor file was
    edited down -- which is how a lost suite was laundered into a green gate.
    """
    proc = run_gate(make_repo(tmp_path, 100, 90, ref="origin"), 500)
    assert proc.returncode == 1, _both(proc)
    assert "gate 3c FAIL (monotonicity)" in proc.stdout
    assert "may only ratchet UP" in proc.stdout
    assert "gate 3c FAIL (count)" not in proc.stdout, "the two arms must be distinct"


def test_a_raised_floor_passes(tmp_path: Path) -> None:
    """Monotonicity is one-directional: the ratchet must not block itself."""
    proc = run_gate(make_repo(tmp_path, 100, 140, ref="origin"), 140)
    assert proc.returncode == 0, _both(proc)


def test_lost_tests_plus_a_lowered_floor_reports_both(tmp_path: Path) -> None:
    """The real-world shape: a suite disappears and the floor is edited to match it.

    Reporting only one of the two would send the author back for a second round, and the
    count arm alone is the arm that a lowered floor silences.
    """
    proc = run_gate(make_repo(tmp_path, 100, 80, ref="origin"), 80)
    assert proc.returncode == 1, _both(proc)
    assert "gate 3c FAIL (count)" in proc.stdout
    assert "gate 3c FAIL (monotonicity)" in proc.stdout


# --- ref resolution -----------------------------------------------------------------------


def test_a_local_dev_is_used_when_no_remote_ref_exists(tmp_path: Path) -> None:
    proc = run_gate(make_repo(tmp_path, 100, 100, ref="local"), 99)
    assert proc.returncode == 1, _both(proc)
    assert "floor is 100 at dev" in proc.stdout


def test_origin_dev_outranks_a_local_dev(tmp_path: Path) -> None:
    """Precedence arm 1 over arm 2: a stale local `dev` must never outrank the fetched one."""
    proc = run_gate(make_repo(tmp_path, 100, 2000, ref="both"), 2000)
    assert proc.returncode == 0, _both(proc)
    assert "floor=100 ref=origin/dev" in proc.stdout, (
        "the local `dev` floor (1100) must not be the one consulted"
    )


def test_a_shallow_checkout_recovers_the_ref_by_fetching(tmp_path: Path) -> None:
    """Arm 3. `actions/checkout@v4` defaults to `fetch-depth: 1`, which leaves a CI job with
    neither `origin/dev` nor local `dev` -- without this arm the ONE run that matters would
    silently take the bootstrap arm."""
    proc = run_gate(make_repo(tmp_path, 100, 100, ref="fetchable"), 99)
    assert proc.returncode == 1, _both(proc)
    assert "floor is 100 at FETCH_HEAD" in proc.stdout


def test_the_bootstrap_arm_is_loud_and_says_what_it_did_not_check(tmp_path: Path) -> None:
    """Arm 4, behaving as documented: it compares, it passes, and it SAYS the property is off.

    The predecessor also had a bootstrap arm. Its defect was that the arm was silent, so a CI
    log could not be told apart from one that had made a real comparison.
    """
    proc = run_gate(make_repo(tmp_path, 100, 90, ref="none"), 95)
    assert proc.returncode == 0, _both(proc)
    assert "BOOTSTRAP ARM" in proc.stdout
    assert "ref=<none:bootstrap>" in proc.stdout
    assert "monotonicity NOT enforced" in _both(proc)
    assert "floor=90" in proc.stdout, "bootstrap compares against the working-tree floor"


def test_the_bootstrap_arm_still_enforces_the_count(tmp_path: Path) -> None:
    proc = run_gate(make_repo(tmp_path, 100, 90, ref="none"), 89)
    assert proc.returncode == 1, _both(proc)
    assert "gate 3c FAIL (count)" in proc.stdout


def test_a_ref_that_predates_the_floor_file_degrades_to_bootstrap(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, 100, 90, ref="local")
    _git(repo, "rm", "-q", "--cached", FLOOR_REL)
    _git(repo, "commit", "-q", "-m", "drop floor")
    _git(repo, "branch", "-q", "-f", "dev", "HEAD")
    _git(repo, "reset", "-q", "HEAD~1")
    _write_floor(repo, 90)
    proc = run_gate(repo, 95)
    assert proc.returncode == 0, _both(proc)
    assert "predates gate 3c" in proc.stderr
    assert "BOOTSTRAP ARM" in proc.stdout


# --- malformed input ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("where", "text"), [("tree", "not-a-number\n"), ("tree", "\n")], ids=["garbage", "empty"]
)
def test_a_floor_file_that_is_not_a_count_is_refused_not_coerced(
    tmp_path: Path, where: str, text: str
) -> None:
    repo = make_repo(tmp_path, 100, 100, ref="origin", name=f"repo-{where}-{len(text)}")
    (repo / FLOOR_REL).write_text(text, encoding="utf-8")
    proc = run_gate(repo, 500)
    assert proc.returncode == 2, _both(proc)
    assert "is not a count" in proc.stderr


def test_an_unknown_argument_is_refused(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, 100, 100, ref="origin")
    proc = subprocess.run(
        ["bash", str(GATE), "--floor", "1"], cwd=repo, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 2
    assert "unknown argument" in proc.stderr


# --- the self-test arms (LAW-07: the trigger proves itself on every invocation) ------------


def test_the_gates_own_self_test_passes() -> None:
    proc = subprocess.run(
        ["bash", str(GATE), "--self-test"], cwd=REPO_ROOT, capture_output=True, text=True,
        check=False,
    )
    assert proc.returncode == 0, _both(proc)
    assert "all correct" in proc.stdout


def _blinded(tmp_path: Path) -> Path:
    """The gate with `verdict` neutered -- the mutation the self-test must catch."""
    text = GATE.read_text(encoding="utf-8")
    broken = text.replace("  return \"$rc\"\n}", "  return 0\n}", 1)
    assert broken != text, "the verdict function's return was not found to mutate"
    path = tmp_path / "blinded_gate.sh"
    path.write_text(broken, encoding="utf-8")
    return path


def test_the_self_test_catches_a_neutered_verdict(tmp_path: Path) -> None:
    proc = subprocess.run(
        ["bash", str(_blinded(tmp_path)), "--self-test"], cwd=REPO_ROOT, capture_output=True,
        text=True, check=False,
    )
    assert proc.returncode == 1, _both(proc)
    assert "SELF-TEST FAIL" in proc.stderr


def test_the_self_test_runs_before_the_real_verdict_not_after(tmp_path: Path) -> None:
    """A self-test that only guards a `--self-test` invocation guards nothing in CI."""
    repo = make_repo(tmp_path, 100, 100, ref="origin")
    proc = run_gate(repo, 500, script=_blinded(tmp_path))
    assert proc.returncode == 1, _both(proc)
    assert "SELF-TEST FAIL" in proc.stderr
    assert "collected=500" not in proc.stdout, "it must abort BEFORE reporting a verdict"


# --- regression pins on the gate's own text ------------------------------------------------


def test_the_gate_no_longer_names_a_branch_this_repo_does_not_have() -> None:
    """The original defect in one assertion: `main` does not exist here, `dev` does."""
    text = GATE.read_text(encoding="utf-8")
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    assert "origin/main" not in code and 'MAIN_BRANCH="main"' not in code
    assert 'MAIN_BRANCH="dev"' in code


# --- F-816-33: a count from an interrupted collection is not a count -----------------------

def test_a_finished_collection_is_measured_and_compared(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, ref_floor=4396, tree_floor=4396, ref="origin")
    proc = run_gate_measuring(repo, _COLLECT_CLEAN)
    assert proc.returncode == 0, _both(proc)
    assert "collected=4414" in proc.stdout


def test_a_broken_collection_is_refused_even_though_it_printed_a_count(tmp_path: Path) -> None:
    """THE defect, end to end: the count parses, clears the floor, and must still not be
    trusted -- 4414 of 4414 collected with one file gone is 4414 of 4428."""
    repo = make_repo(tmp_path, ref_floor=4396, tree_floor=4396, ref="origin")
    proc = run_gate_measuring(repo, _COLLECT_BROKEN)
    assert proc.returncode != 0, _both(proc)
    assert "FAIL (collection)" in _both(proc)
    assert "exited 2" in _both(proc)


def test_the_failing_module_is_named_in_the_refusal(tmp_path: Path) -> None:
    """A refusal that does not carry pytest's own error block sends the reader back to re-run
    the collection by hand, which is where the predecessor's silence started."""
    repo = make_repo(tmp_path, ref_floor=4396, tree_floor=4396, ref="origin")
    assert "test_planted.py" in _both(run_gate_measuring(repo, _COLLECT_BROKEN))


def test_a_swallowed_status_is_caught_by_the_summary_line(tmp_path: Path) -> None:
    """The second arm of `collection_verdict`, and not redundant: the predecessor's bug WAS a
    lost exit status, so a guard that only reads the status would have been unreachable there."""
    repo = make_repo(tmp_path, ref_floor=4396, tree_floor=4396, ref="origin")
    proc = run_gate_measuring(repo, _COLLECT_STATUS_SWALLOWED)
    assert proc.returncode != 0, _both(proc)
    assert "reports collection errors" in _both(proc)


def test_an_injected_collection_command_announces_itself(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, ref_floor=4396, tree_floor=4396, ref="origin")
    proc = run_gate_measuring(repo, _COLLECT_CLEAN)
    assert "collection command INJECTED via --pytest-cmd" in proc.stdout


def test_no_ci_step_injects_a_count() -> None:
    """`--collected` is a test seam. In a CI `run:` body it would be a way to fake the gate."""
    workflow = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    body = workflow.read_text(encoding="utf-8")
    assert "--collected" not in body
    assert "--pytest-cmd" not in body, "the collection seam would fake the gate the same way"


def test_the_committed_tree_is_green_against_the_real_ref() -> None:
    """R98: a gate may only be adopted over a clean baseline. Uses the tree's OWN floor as the
    injected count, which proves the ref lookup and both comparisons run end to end without
    paying for a real collection."""
    floor = int((REPO_ROOT / FLOOR_REL).read_text(encoding="utf-8").strip())
    proc = run_gate(REPO_ROOT, floor)
    assert proc.returncode == 0, _both(proc)
    assert f"tree_floor={floor}" in proc.stdout
