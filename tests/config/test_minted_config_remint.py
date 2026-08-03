"""⊕ WPMAIN ORACLE — the re-mint is ADDITIVE, and run5's armed values survive it (O-E2).

This WP adds three items to the schema — `eval_enabled` (R120), the `monitor.disk_guard`
family (R122) and `train.device` (R126) — and re-mints all six configs by replaying each
one's header-recorded template + deltas. A replay is a REGENERATION: it can silently move any
value the template owns. `configs/run5.yaml` carries the three pre-registered armed values
`0.25 / 25000 / 50` (`train.draw_rate_abort`), and R119's HARD STOP is verbatim: **any change
to run5's armed values** is mint-prereg-only. This file is that stop's instrument.

The baseline is committed, not computed: `tests/fixtures/wpmain/config_baseline_b482243/`
holds the six configs' bytes at `b482243`, the commit this WP branched from, with manifest
rows and sha256s (`tests/fixtures/manifest.toml`). A golden that lives in git history alone
is a golden no test can read.

Two independent instruments, because they see different mutations:

- **structural** — the parsed leaf key-path -> value map. Every pre-existing leaf keeps its
  value byte-for-byte, exactly the new leaves appear, and exactly the RULED-DELETED leaf
  disappears. This sees a moved VALUE regardless of formatting.
- **textual** — the line-level diff must be INSERT-ONLY apart from that one ruled deletion.
  This sees a dropped minted-header line, a reordered block, a rewritten comment: things a
  value map is blind to, and things a regenerating minter does easily.

WHY THERE IS A DELETION AT ALL, AND WHY THE BASELINE WAS NOT RE-CUT (WP12-R, R178(a) as
assigned by R183(a)). R178(a) DELETES `train.buffer_save_interval` — a key minted into
`run5.yaml` whose only consumer chain ended in `_try_save_buffer`, which WP12-R Phase CS
(F-CS-2) measured production-dead on every leg. The re-mint that rides that deletion is the
first NON-insertion this file has ever seen, and closing it had exactly two shapes:

1. **Re-cut the baseline.** REJECTED, on three grounds and one of them is dispositive.
   (a) `tests/fixtures/manifest.toml` — which carries each baseline file's sha256 and is what
   `test_fixtures_manifest.py` checks — is a FROZEN oracle in this WP; re-cutting the baseline
   means editing it, and no grant exists. (b) The directory name `config_baseline_b482243` IS
   a commit pin; a baseline re-cut against a later tree makes its own name false. (c) A re-cut
   baseline equals the live configs, so `_ADDED_LEAVES` becomes empty and every assertion in
   this file goes vacuous — that is destroying the instrument, not maintaining it.
2. **Teach the instrument that ONE ruled deletion is legal, by name.** ADOPTED. `_REMOVED_
   LEAVES` is a closed, one-element set carrying its R-number, and both halves assert set
   EQUALITY against it rather than relaxing to "deletions are fine": a second dropped leaf, a
   dropped header line, a reformat, a reorder and a rewritten comment all still red. The
   textual half additionally pins the exact deleted TEXT and forbids `replace` outright, so a
   deletion cannot smuggle a substitution in beside it.

The instrument's reach is therefore unchanged except on one named line. Anything the
pre-R178 version caught, this version still catches — asserted directly by
`test_the_permitted_deletion_is_exactly_one_named_line`, which fails if the allowance ever
widens beyond that one leaf.

Device VALUES are deliberately NOT pinned per config. R126 routes them to mint prereg
(DEP-DEVICE-MINT-VALUES): `smoke_preflight_armed` is the one unambiguous case — its own
header already records `eval.worker_device: cuda -> cpu` and it is the CI-runnable smoke — so
cpu is transcription there and IS pinned. The other five are the operator's decision at mint,
and an oracle that pinned them would be this phase making a mint decision.

The six configs are ENUMERATED here rather than read from `tests/conftest.py:52-53`'s
`MINTED_CONFIGS`, which lists FIVE — it omits `smoke_preflight_armed.yaml`. That gap is
pre-existing and recorded, not fixed here (F-12); inheriting it would leave the config the
preflight actually boots out of the HARD-STOP sweep.
"""
from __future__ import annotations

import difflib
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_LIVE = _REPO / "configs"
_BASELINE = _REPO / "tests" / "fixtures" / "wpmain" / "config_baseline_b482243"

_CONFIGS = ("dev_example.yaml", "run5.yaml", "smoke_gnn.yaml", "smoke_preflight_armed.yaml",
            "smoke_radius_curriculum.yaml", "sustained_kcluster.yaml")

#: Exactly what this WP's re-mint may add — one key, one family of three leaves, one key.
_ADDED_LEAVES = {
    "eval_enabled",
    "monitor.disk_guard.interval_sec",
    "monitor.disk_guard.warn_gb",
    "monitor.disk_guard.fail_gb",
    "train.device",
}

#: Exactly what a re-mint may REMOVE, ever — the one dead knob R178(a) ORDERED deleted
#: (WP12-R, assigned to its dispatcher by R183(a); grounds R116/LAW-08 + the F-CS-2
#: measurement that the replay-buffer save is production-dead on every leg). A closed set of
#: one. Widening it is a ruling, not an edit.
_REMOVED_LEAVES = {"train.buffer_save_interval"}

#: The one deleted LINE, byte-exact, that the textual half will tolerate. Every minted config
#: writes this leaf identically (`yaml.safe_dump`, two-space indent, value `0`), so pinning
#: the text costs nothing and buys the guarantee that the tolerated deletion is the ruled one
#: and not merely a deletion of the same SIZE somewhere else in the file.
_REMOVED_LINES = ["  buffer_save_interval: 0"]

#: R122's minted family (revisable at mint prereg per the R85 pattern — the literals were
#: dead, so nothing has ever measured them; a revision is a prereg row, not an IMPL edit).
_DISK_GUARD = {"monitor.disk_guard.interval_sec": 60.0,
               "monitor.disk_guard.warn_gb": 10.0,
               "monitor.disk_guard.fail_gb": 5.0}

#: R119's HARD STOP, in the config's own units.
_RUN5_ARMED = {"train.draw_rate_abort.threshold": 0.25,
               "train.draw_rate_abort.min_step": 25000,
               "train.draw_rate_abort.N_pool_min": 50}


def _leaves(node, prefix: str = "") -> dict[str, object]:
    """Every leaf key-path -> value. A list is a LEAF: reordering a ladder's rungs is a
    value change, not a structural one, and must be seen as such."""
    if isinstance(node, dict):
        out: dict[str, object] = {}
        for key, value in node.items():
            out.update(_leaves(value, f"{prefix}{key}."))
        return out
    return {prefix.rstrip("."): node}


def _live_leaves(name: str) -> dict[str, object]:
    return _leaves(yaml.safe_load((_LIVE / name).read_text(encoding="utf-8")))


def _baseline_leaves(name: str) -> dict[str, object]:
    return _leaves(yaml.safe_load((_BASELINE / name).read_text(encoding="utf-8")))


def test_the_committed_baseline_covers_every_minted_config() -> None:
    """Premise — an oracle whose golden is missing is an oracle that passes vacuously. The
    fixtures manifest holds the sha256 of each of these six (added_by = "WPMAIN"), so a
    baseline edited to make a diff go away is caught by `test_fixtures_manifest.py`, not by
    nobody."""
    assert sorted(path.name for path in _BASELINE.glob("*.yaml")) == sorted(_CONFIGS)
    assert sorted(path.name for path in _LIVE.glob("*.yaml")) == sorted(_CONFIGS), (
        "the live set must still be the six this WP re-mints — a seventh config appearing "
        "mid-WP needs its own baseline row before this sweep means anything"
    )


@pytest.mark.parametrize("name", _CONFIGS)
def test_the_remint_adds_the_new_keys_and_moves_nothing_else(name: str) -> None:
    """O-E2, structural half — set equality on what appeared, and value equality on
    everything that was already there.

    MUTATION THAT REDS IT: a replay that regenerates from a drifted template and moves,
    say, `train.batch_size` or an eval ladder rung. That is invisible in review (the diff is
    "just a re-mint") and it silently re-postures every run. Value equality over ~170 leaves
    is the only instrument that sees it."""
    baseline, live = _baseline_leaves(name), _live_leaves(name)
    removed = set(baseline) - set(live)
    assert removed == _REMOVED_LEAVES, (
        f"{name}: a re-mint may drop exactly {sorted(_REMOVED_LEAVES)} (R178(a), the one "
        f"ruled dead-knob deletion); got {sorted(removed)}"
    )
    added = set(live) - set(baseline)
    assert added == _ADDED_LEAVES, (
        f"{name}: the re-mint may add exactly {sorted(_ADDED_LEAVES)} (R120 + R122 + R126); "
        f"got {sorted(added)}"
    )
    moved = {path: (baseline[path], live[path])
             for path in baseline
             if path not in _REMOVED_LEAVES and live[path] != baseline[path]}
    assert not moved, (
        f"{name}: the re-mint MOVED existing values {moved} — the three schema items are "
        "additive by construction (§6), so any other delta is template drift"
    )


@pytest.mark.parametrize("name", _CONFIGS)
def test_the_remint_diff_is_insert_only_apart_from_the_one_ruled_deletion(name: str) -> None:
    """O-E2, textual half — nothing is rewritten, including the minted header, and the ONLY
    thing deleted is R178(a)'s named line.

    The header records `minted-by`, `template` and every delta (`configs/run5.yaml:1-7`); it
    is the provenance a mint record is reconstructed from. A value map cannot see a lost
    header line, a reordered section, or a rewritten comment.

    `replace` stays forbidden outright, and that is what keeps the allowance narrow: a
    reformat presents as `replace`, so tolerating `delete` does not tolerate a rewrite. A
    `delete` is tolerated only when its removed lines are EXACTLY `_REMOVED_LINES` — same
    text, same count — so a second dropped line, a dropped header line, or a different line
    of the same length all still red.

    MUTATION THAT REDS IT: a minter that reformats (re-wraps a long ladder line, re-orders
    keys, normalises quoting). The diff stops being insert-only, and the re-mint stops being
    reviewable as "the new keys, minus the one ruled deletion, and nothing else"."""
    baseline = (_BASELINE / name).read_text(encoding="utf-8").splitlines()
    live = (_LIVE / name).read_text(encoding="utf-8").splitlines()
    ops = difflib.SequenceMatcher(a=baseline, b=live, autojunk=False).get_opcodes()
    offending = [
        (tag, baseline[i1:i2], live[j1:j2])
        for tag, i1, i2, j1, j2 in ops
        if not (tag in ("equal", "insert")
                or (tag == "delete" and baseline[i1:i2] == _REMOVED_LINES))
    ]
    assert not offending, (
        f"{name}: the re-mint diff must be INSERT-ONLY apart from the single ruled deletion "
        f"{_REMOVED_LINES} (R178(a)); found {offending[:3]}"
    )
    deleted = [line for tag, i1, i2, _, _ in ops if tag == "delete" for line in baseline[i1:i2]]
    assert deleted == _REMOVED_LINES, (
        f"{name}: exactly one line may be deleted, once — R178(a)'s; got {deleted}"
    )


def test_the_permitted_deletion_is_exactly_one_named_line() -> None:
    """The allowance's own guard: this file tolerates deletion at all only because R178(a)
    ORDERED one, and the tolerance must never become "deletions are fine".

    MUTATION THAT REDS IT: a later phase that needs its own key gone and widens
    `_REMOVED_LEAVES` / `_REMOVED_LINES` instead of getting a ruling. The sets are closed at
    one element each and the leaf and the line must name the SAME key, so a widened allowance
    cannot pass as a maintenance edit."""
    assert _REMOVED_LEAVES == {"train.buffer_save_interval"}, (
        "the ruled deletion set is R178(a)'s and is closed at one leaf; widening it is a "
        "ruling (R183(a) assigned this one deletion, not a deletion policy)"
    )
    assert _REMOVED_LINES == ["  buffer_save_interval: 0"], (
        "the tolerated line must be the ruled leaf's own minted line"
    )
    assert not (_REMOVED_LEAVES & _ADDED_LEAVES), (
        "a leaf cannot be both added and removed by one re-mint"
    )


@pytest.mark.parametrize("name", _CONFIGS)
def test_every_config_mints_the_disk_guard_family_at_the_ruled_values(name: str) -> None:
    """O-E2, the R122 family arm. The three values are the previously-DEAD code-side
    literals (`subsystems.py:163-168`), minted so the guard R121(b) mandates has operator-
    visible thresholds instead of numbers nobody could see.

    MUTATION THAT REDS IT: mint a different value on one config (a per-config guard posture
    nobody decided), or mint the block on five of six."""
    leaves = _live_leaves(name)
    for path, value in _DISK_GUARD.items():
        assert leaves[path] == value, (
            f"{name}: {path} must mint at R122's ruled {value}; got {leaves[path]!r}"
        )
    assert leaves["eval_enabled"] is True, (
        f"{name}: eval_enabled mints True everywhere — today's effective posture is the "
        "code default True, so True is a zero-behaviour mint (R120; run5's True is LAW-15)"
    )


@pytest.mark.parametrize("name", _CONFIGS)
def test_every_config_declares_a_train_device_from_the_closed_vocabulary(name: str) -> None:
    """O-E2, the R126 arm — the KEY is pinned on all six; the VALUE is pinned only where the
    design states it.

    `smoke_preflight_armed` = cpu is transcription, not a decision: its own minted header
    records `eval.worker_device: cuda -> cpu`, every in-repo drive ran it on cpu, and it is
    the CI-runnable smoke (§C.1.6). The other five are operator mint decisions
    (DEP-DEVICE-MINT-VALUES) — pinning them here would be this phase deciding run5's device,
    which is not its call.

    MUTATION THAT REDS IT: transcribe `eval.worker_device` into `train.device` across the
    board. R126 rules them DIFFERENT facts (split topology legitimate), so that inference is
    exactly the proxy the ruling refuses — and it would silently mint the armed smoke to
    cuda, un-CI-running the one config the preflight boots."""
    device = _live_leaves(name)["train.device"]
    assert device in ("cpu", "cuda"), (
        f"{name}: train.device must be a member of the closed vocabulary; got {device!r}"
    )
    if name == "smoke_preflight_armed.yaml":
        assert device == "cpu", (
            "the armed smoke is a genuinely-cpu smoke because its CONFIG is cpu, not because "
            "a flag said so (R64/R61, §C.1.2)"
        )


def test_run5s_armed_draw_rate_values_survive_the_remint_byte_identical() -> None:
    """O-E2's HARD-STOP arm (R119, verbatim: "any change to run5's armed values — 0.25 /
    25000 / 50 — mint-prereg-only").

    Asserted THREE ways, because each sees a different way to lose them: the parsed values
    (a changed number), the literal source lines (a reformat that rounds `0.25` to `0.3`
    through a float round-trip), and the minted header's delta line (a replay that dropped
    the delta and fell back to the template's `None`, which DISARMS the abort entirely while
    every schema check stays green).

    MUTATION THAT REDS IT: any of the three. This is the one assertion in the WP that is not
    about correctness but about authority: these numbers were pre-registered at mint prereg
    (R82/R85/R92) and no work package may move them."""
    leaves = _live_leaves("run5.yaml")
    for path, value in _RUN5_ARMED.items():
        assert leaves[path] == value, (
            f"run5 {path} is {leaves[path]!r}, not the pre-registered {value!r} — HARD STOP"
        )
    text = (_LIVE / "run5.yaml").read_text(encoding="utf-8")
    for line in ("    threshold: 0.25", "    min_step: 25000", "    N_pool_min: 50"):
        assert line in text, f"run5's armed line {line!r} is not byte-identical after re-mint"
    assert ("# delta: train.draw_rate_abort: None -> {'threshold': 0.25, 'min_step': 25000, "
            "'N_pool_min': 50, 'consec': 3}") in text, (
        "the minted header's own record of the arming delta must survive the replay — "
        "without it the provenance of the armed values is gone even if the values are not"
    )
