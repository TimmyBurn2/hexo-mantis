"""⊕ WPAX Phase P ORACLE — C-4: the armed-abort manifest (DESIGN_P §8, §13.2).

RED-at-import until IMPL lands `mantis.config.armed_aborts` (`Status`, `Mechanism`,
`ArmedAbort`, `MANIFEST`, `PRODUCTION_CONFIGS`, `audit_arming`) and
`tools/ci_gates/preflight_mint.py` (`verify_source_pins`, which SF-4 moves OUT of the
shipped package). Both loads below are the RED anchor; every oracle in this file rides on
them, exactly as `tests/test_run_strict_composition.py` rides on
`mantis.config.resolve.composition`.

What this file exists to stop, in one sentence: run5 minting with a required abort silently
disarmed, or with the DEFERRED draw-rate row quietly rotting into the status quo.

The oracles, and the defect each one is the ONLY witness to:

- M1  `test_arming_audit_fails_a_disarmed_production_config` — a `required` row disarmed in
      a real committed config is caught. Sole witness to assertion (c) biting at all.
- M1' `test_the_audit_reads_the_CONFIG_not_the_config_FILENAME` — the cheapest passing
      implementation of M1 is `disarmed iff path endswith dev_example.yaml`. Only this
      oracle kills it: run5 with the value flipped to False must be disarmed, and
      dev_example with it flipped to True must not.
- M10 `test_the_deferred_row_source_pin_is_tamper_evident` — R56's asymmetry: a pin that
      matches NOTHING is itself a hard failure. Sole witness to the forcing function that
      makes Phase D's DEFERRED→required flip unforgettable (§8.4).
- O-6  `test_the_manifest_is_not_vacuous` — gate 11's `MIN_SCANNED_FILES` floor
      (`silent_encoding_gate.py:70,331-336`) transplanted. An empty manifest audits every
      config green; sole witness to that.
- O-6b `test_row_invariants_are_enforced_at_construction` — `__post_init__` (§8.2). Sole
      witness that `status` and `owner`/`source_pin` cannot drift apart.
- O-6c `test_the_manifest_module_makes_no_filesystem_call` — SF-4's layer boundary. Sole
      witness; without it the shipped package silently re-acquires repo-root knowledge that
      works only under an editable install.
- O-7  `test_flipping_the_deferred_row_to_required_needs_no_code_change` — §8.5, proven not
      asserted. Sole witness that Phase D's flip is a DATA edit.
- O-11 `test_a_minted_config_survives_a_dump_revalidate_round_trip` — SF-10. The burst
      override's central premise (§5.3), sound at HEAD only by accident. Sole witness: a
      future `alias` / `computed_field` / `exclude=` / serializer would otherwise first
      surface inside the mint preflight on the training box.

R7 / gate 6: nothing here writes a `*.jsonl`, and the M10 tamper rig is built under
`tmp_path`, never in the tree.

>300 justify (R8): the manifest has exactly one consumer surface and this is it — the
audit predicate, the row invariants, the tamper scan and the DEFERRED→required flip
simulation all read the SAME `MANIFEST` rows through the same `_required`/`_deferred`/
`_dotted` spine, and O-11 is the premise the tool's burst override rests on. Splitting
would fork that spine across files (R5 bars cross-test imports). The length is prose, not
logic: roughly half of it is the per-oracle "what is this the only witness to" rationale
LAW-07 requires each row to carry.
"""
from __future__ import annotations

import importlib.util
import tokenize
from pathlib import Path
from types import SimpleNamespace

import pytest
from mantis.config.armed_aborts import (  # RED-at-import anchor: module absent at HEAD
    MANIFEST,
    PRODUCTION_CONFIGS,
    ArmedAbort,
    Mechanism,
    Status,
    audit_arming,
)
from mantis.config.loader import load_config
from mantis.config.schema import RunConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "ci_gates" / "preflight_mint.py"
MANIFEST_MODULE = REPO_ROOT / "src" / "mantis" / "config" / "armed_aborts.py"
CONFIG_PATHS = tuple(sorted(p.name for p in (REPO_ROOT / "configs").glob("*.yaml")))


def _load_tool():
    """Load the gate script by absolute path — the `tests/tools/test_silent_encoding_gate.py
    :21-29` precedent, ZERO `sys.path` mutation (R5 / LAW-17). `tools/` is not a package."""
    if not TOOL_PATH.is_file():
        raise ModuleNotFoundError(
            f"RED anchor: {TOOL_PATH.relative_to(REPO_ROOT)} does not exist at HEAD — "
            "IMPL owes C-2 (DESIGN_P §13). SF-4 places `verify_source_pins` and ALL "
            "repo-root path resolution here, not in the shipped package."
        )
    spec = importlib.util.spec_from_file_location("preflight_mint", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()  # RED-at-import anchor #2


def _dotted(obj, path: str):
    """Walk a dotted config path. Deliberately reimplemented here rather than importing the
    module's own helper: an oracle that navigates with the code under test cannot witness a
    navigation bug."""
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _code_text(path: Path) -> str:
    """Source with COMMENT / STRING / f-string-literal tokens removed.

    A census over raw text would flag the module's own prose ("this module makes no
    filesystem call"), which is the false positive that teaches people to word documents
    around a gate. Tokenizing is the honest form.
    """
    skip = {tokenize.COMMENT, tokenize.STRING, tokenize.FSTRING_MIDDLE}
    with path.open("rb") as handle:
        return "\n".join(
            tok.string for tok in tokenize.tokenize(handle.readline) if tok.type not in skip
        )


def _required(manifest=MANIFEST):
    return [row for row in manifest if row.status is Status.REQUIRED]


def _deferred(manifest=MANIFEST):
    return [row for row in manifest if row.status is Status.DEFERRED]


# ── M1 — assertion (c) bites ──────────────────────────────────────────────────────────
def test_arming_audit_fails_a_disarmed_production_config() -> None:
    """`configs/dev_example.yaml` is a REAL committed file with
    `actor_lag_abort_enabled: false` (`:200`), so this needs no mutation at all — the
    corpus row is the tree. The inverse arm (run5, armed at `:203` since R59) is asserted in
    the same test because a gate that only ever says FAIL is as useless as one that only
    ever says PASS."""
    disarmed = audit_arming(load_config(REPO_ROOT / "configs" / "dev_example.yaml"))
    assert [row.name for row in disarmed.disarmed] == ["actor_lag", "draw_rate_collapse"], (
        "a production config with BOTH hard aborts disarmed must name those rows and only "
        "those rows. WPAX Phase D (R87 hunk 1): `dev_example.yaml` ships "
        "`actor_lag_abort_enabled: false` at `:200` AND `train.draw_rate_abort: null` — the "
        "second is R59's deliberate smoke disarm, made observable by the `null` spelling "
        "instead of inferable from an absent key. The 'and only those rows' bite is what "
        f"this expectation keeps; got {[row.name for row in disarmed.disarmed]}"
    )
    assert [row.config_path for row in disarmed.disarmed] == [
        "monitor.actor_lag_abort_enabled", "train.draw_rate_abort.threshold"
    ], "each disarmed row must carry its dotted arming surface, so the report can name it"

    armed = audit_arming(load_config(REPO_ROOT / "configs" / "run5.yaml"))
    assert list(armed.disarmed) == [], (
        "configs/run5.yaml arms the actor-lag abort at `:203` (the R59 flip) — mode AUDIT "
        f"must be GREEN on it today; got {[row.name for row in armed.disarmed]}"
    )
    assert [row.name for row in armed.required] == [row.name for row in _required()], (
        "the audit's `required` list is the manifest's, unfiltered — it is what the report "
        "publishes as `required_armed`"
    )


def test_the_audit_reads_the_CONFIG_not_the_config_FILENAME(smoke_run_config) -> None:
    """The cheapest implementation that passes M1 is a filename check. Both arms of this
    oracle are needed to kill it: the two configs swap verdicts when — and only when — the
    VALUE swaps. Driven through the blessed `load_config → model_dump → model_validate`
    factory (`tests/conftest.py:67-81`), so both payloads are schema-valid."""
    run5_disarmed = smoke_run_config("run5.yaml", monitor={"actor_lag_abort_enabled": False})
    assert [row.name for row in audit_arming(run5_disarmed).disarmed] == ["actor_lag"], (
        "run5 with the arming flipped OFF must fail the audit — the audit reads the "
        "validated config object, never the path it came from"
    )
    dev_armed = smoke_run_config(
        "dev_example.yaml",
        monitor={"actor_lag_abort_enabled": True},
        # WPAX Phase D (R87 hunk 2): the manifest now carries TWO required rows, so "the
        # arming flipped ON" means flipping BOTH postures. Leaving the draw-rate block
        # `null` here would leave the config disarmed on a row this arm expects nothing
        # from, and the oracle's real subject — value-not-filename — would be lost behind an
        # unrelated red. The values are the run5 prereg ones; `min_step` is inside
        # dev_example's own `max_train_steps`, which the twin cross-validator requires.
        train={"draw_rate_abort": {"threshold": 0.25, "min_step": 25000,
                                   "min_samples": 50}},
    )
    assert list(audit_arming(dev_armed).disarmed) == [], (
        "dev_example with BOTH armings flipped ON must PASS — otherwise the audit is keyed "
        "on the filename and assertion (c) is decoration"
    )


# ── M10 — R56 tamper-evidence ─────────────────────────────────────────────────────────
def test_the_deferred_row_source_pin_is_tamper_evident(tmp_path) -> None:
    """`source_pin` is `(repo-relative path, exact source text)` and the scan asserts the
    string is still THERE. R56's asymmetry (`silent_encoding_gate.py:338-344`): a pin that
    matches nothing is a HARD failure, not a quiet pass — so when Phase D deletes
    `draw_rate_threshold: float = 0.0` (R65), this goes red and the row's flip cannot be
    forgotten.

    Three arms, because only the three together pin the asymmetry: the real tree is clean,
    a tree with the text REMOVED is dirty, and a tree with the FILE removed is dirty too
    (a vanished pin must not read as 'nothing to check').
    """
    pinned = [row for row in MANIFEST if row.source_pin is not None]
    assert pinned, "at least one row must carry a source_pin, or M10 has no subject"

    assert list(TOOL.verify_source_pins(MANIFEST, repo_root=REPO_ROOT)) == [], (
        "every pinned source text must still be present in the real tree — a pin that has "
        "already rotted at oracle-write time is a manifest bug, not a Phase D signal"
    )

    row = pinned[0]
    rel, text = row.source_pin
    original = (REPO_ROOT / rel).read_text()
    assert text in original, f"the pin {text!r} must exist in {rel} at HEAD"

    tampered_root = tmp_path / "tampered"
    (tampered_root / rel).parent.mkdir(parents=True)
    (tampered_root / rel).write_text(original.replace(text, "# pinned literal deleted\n"))
    assert [broken.name for broken in TOOL.verify_source_pins(
        MANIFEST, repo_root=tampered_root)] == [row.name], (
        f"deleting the pinned literal from {rel} must report exactly the {row.name!r} row "
        "as broken — this is the Phase D forcing function (§8.4)"
    )

    absent_root = tmp_path / "absent"
    absent_root.mkdir()
    assert [broken.name for broken in TOOL.verify_source_pins(
        MANIFEST, repo_root=absent_root)] == [row.name], (
        "a pinned file that does not exist must be reported broken, never skipped — "
        "'nothing to scan' is how a tamper-evidence gate goes silently vacuous"
    )


# ── O-6 — the vacuity floor and the row invariants ────────────────────────────────────
def test_the_manifest_is_not_vacuous() -> None:
    """An empty manifest audits every config green, and every downstream assertion about
    (c) becomes true by having nothing to say. This is `MIN_SCANNED_FILES`
    (`silent_encoding_gate.py:70`) applied to the manifest, plus the R4/LAW-08 half: every
    row must name something that EXISTS."""
    assert len(_required()) >= 1, (
        "the manifest must carry at least one `required` row — with none, assertion (c) "
        "passes vacuously on a config with every abort disarmed"
    )
    assert len(PRODUCTION_CONFIGS) >= 1, (
        "PRODUCTION_CONFIGS must name at least one config — it is the single authority for "
        "WHICH configs the law binds (R59 expresses smoke exemption by ABSENCE from it)"
    )
    for rel in PRODUCTION_CONFIGS:
        assert (REPO_ROOT / rel).is_file(), (
            f"PRODUCTION_CONFIGS names {rel!r}, which does not exist. The tuple holds "
            "repo-relative STRINGS (data); resolving them is the tool's (SF-4)"
        )
    assert all(isinstance(row, ArmedAbort) for row in MANIFEST)

    run5 = load_config(REPO_ROOT / "configs" / "run5.yaml")
    for row in _required():
        assert _dotted(run5, row.config_path) is not None, (
            f"required row {row.name!r} names {row.config_path!r}, which does not resolve "
            "on a real RunConfig — a manifest row whose arming surface does not exist is a "
            "phantom gate input (R4 / LAW-07)"
        )
    # WPAX Phase D (R87 hunk 3 — THE VACUOUS ONE). This loop iterated `_deferred()`, and
    # after the flip the shipped manifest holds ZERO deferred rows: it stopped failing by
    # stopping to say anything, which is worse than red. R87 requires it re-pointed to a live
    # subject or deleted with grounds, and the subject is live — the DEFERRED machinery
    # SURVIVES because CARD-COORD-KNOBS (R78/R80) will feed it rows, and `_print_deferred_rows`
    # was kept for exactly that reason (R81). So the rule "a deferred row's pin must name a
    # file that exists" is driven on a synthetic row through `_deferred`'s own `manifest`
    # parameter — the same seam `audit_arming` exposes — and the post-flip fact is asserted
    # rather than assumed.
    assert _deferred() == [], (
        "the shipped manifest holds ZERO deferred rows after Phase D's flip; a row kept "
        "deferred so this loop had a subject was REJECTED (R81) — the manifest is a "
        f"mint-read artifact and does not assert dead deferrals (R87); got {_deferred()}"
    )
    probe = ArmedAbort(
        name="_synthetic_deferred_probe", config_path="train.does_not_exist",
        mechanism=Mechanism.CONFIG_BOOL, status=Status.DEFERRED, exit_code=None,
        owner="CARD-COORD-KNOBS (R78)",
        source_pin=("src/mantis/config/armed_aborts.py", "class ArmedAbort"),
        note="synthetic subject for the deferred-row invariants; not a shipped row.",
    )
    deferred_rows = _deferred((*MANIFEST, probe))
    assert [row.name for row in deferred_rows] == [probe.name], (
        "`_deferred` selects on `status` and nothing else — a selector that branched on a "
        f"row's name or returned a constant is what this arm refuses; got {deferred_rows}"
    )
    for row in deferred_rows:
        rel, _text = row.source_pin
        assert (REPO_ROOT / rel).is_file(), (
            f"deferred row {row.name!r} pins {rel!r}, which does not exist"
        )


def test_row_invariants_are_enforced_at_construction() -> None:
    """§8.2's `__post_init__`. `status` selects the list and `mechanism` selects the
    predicate — both DATA — so the one thing that can rot is a row whose status and
    ownership disagree. All three arms are driven because each is a different way for the
    DEFERRED row to become invisible: an owner-less deferred row has nobody to chase, a
    pin-less one is not tamper-evident, and a required row carrying an owner reads as
    already-excused."""
    common = dict(name="probe", config_path="monitor.actor_lag_abort_enabled",
                  mechanism=Mechanism.CONFIG_BOOL, exit_code=None, note="oracle probe")
    pin = ("src/mantis/train/coordinator/config.py", "draw_rate_threshold: float = 0.0")

    with pytest.raises(ValueError):
        ArmedAbort(status=Status.DEFERRED, owner=None, source_pin=pin, **common)
    with pytest.raises(ValueError):
        ArmedAbort(status=Status.DEFERRED, owner="SOMEBODY", source_pin=None, **common)
    with pytest.raises(ValueError):
        ArmedAbort(status=Status.REQUIRED, owner="SOMEBODY", source_pin=None, **common)

    # …and the two legal shapes construct, so the invariant is not simply "always raise".
    ArmedAbort(status=Status.REQUIRED, owner=None, source_pin=None, **common)
    ArmedAbort(status=Status.DEFERRED, owner="SOMEBODY", source_pin=pin, **common)


def test_the_manifest_module_makes_no_filesystem_call() -> None:
    """SF-4's layer boundary, pinned where it can rot. `parents[3]` resolves to the repo
    root ONLY because this install is editable; a wheel-installed `mantis` would resolve it
    into site-packages. The data stays in the package; every `Path`, `read_text` and
    repo-root resolution lives in the tool (`REPO_ROOT = Path(__file__).resolve().parents[2]`
    — `silent_encoding_gate.py:62`, where the idiom is structurally sound)."""
    code = _code_text(MANIFEST_MODULE)
    for token in ("__file__", "pathlib", "Path(", "read_text", "open(", "os.path",
                  "glob(", "rglob(", "iterdir", "exists("):
        assert token not in code, (
            f"src/mantis/config/armed_aborts.py must make no filesystem call; found "
            f"{token!r}. SF-4: the shipped package may not carry repo-root knowledge"
        )


# ── O-7 — Phase D's flip is a DATA edit ───────────────────────────────────────────────
def test_flipping_the_deferred_row_to_required_needs_no_code_change() -> None:
    """§8.5, proven rather than asserted. Builds an in-memory manifest with the draw-rate
    row flipped to REQUIRED (owner and source_pin dropped, which `__post_init__` then
    demands) and a config carrying the key Phase D will add, and drives `audit_arming`
    unchanged. If Phase D ever DOES need a code change, this test says so before it starts.

    Both threshold arms are driven: `CONFIG_THRESHOLD_GT_ZERO` must route through the same
    branch-free path as `CONFIG_BOOL` (`audit_arming` never branches on `name`), and a
    mechanism that returned a constant would pass only one of the two.
    """
    # WPAX Phase D (R87 hunk 4). Phase D HAS landed, so the shipped manifest holds zero
    # deferred rows and `len(deferred) == 1` reds. What this test proves survives the flip
    # unchanged and is the reason it is kept rather than deleted: `audit_arming` DISPATCHES
    # ON DATA — `status` selects the list, `mechanism` selects the predicate — and no
    # function branches on a row's name. That claim needs a row to flip, so the subject is
    # synthetic and the drive is otherwise identical.
    assert _deferred() == [], (
        "the flip is landed: the shipped manifest carries no deferred row (R87's own "
        f"grounds — a mint-read artifact does not assert dead deferrals); got {_deferred()}"
    )
    row = ArmedAbort(
        name="draw_rate_collapse", config_path="train.draw_rate_abort.threshold",
        mechanism=Mechanism.CONFIG_THRESHOLD_GT_ZERO, status=Status.DEFERRED,
        exit_code=None, owner="CARD-COORD-KNOBS (R78)",
        source_pin=("src/mantis/run.py", "def compose_run"),
        note="synthetic pre-flip subject; the shipped row is REQUIRED since Phase D.",
    )
    assert [other.name for other in _deferred((*MANIFEST, row))] == [row.name], (
        "the DEFERRED machinery still works and still selects on status alone — that is what "
        "makes the flip below a DATA edit rather than a code change"
    )
    flipped = ArmedAbort(
        name=row.name, config_path=row.config_path, mechanism=row.mechanism,
        status=Status.REQUIRED, exit_code=row.exit_code, owner=None, source_pin=None,
        note=row.note,
    )
    manifest = tuple(flipped if other.name == row.name else other for other in MANIFEST)

    def _future_config(threshold: "float | None"):
        """The shape Phase D's schema extension DID produce. A stub, not a RunConfig — the
        `RunConfig` drive is `tests/config/test_drawrate_arming_authority.py`'s O-D3, which
        exists because a stub built FROM `config_path` cannot disagree with it."""
        return SimpleNamespace(
            monitor=SimpleNamespace(actor_lag_abort_enabled=True),
            train=SimpleNamespace(
                draw_rate_abort=None if threshold is None
                else SimpleNamespace(threshold=threshold)),
        )

    off = audit_arming(_future_config(None), manifest=manifest)
    assert [r.name for r in off.disarmed] == [row.name], (
        "with the row flipped to `required` and the block EXPLICITLY disarmed, the audit "
        "must report it disarmed. N-c: the off arm is `None`, not `0.0` — under `gt=0, le=1` "
        "the system can no longer PRODUCE `0.0`, so an assertion over it would assert "
        f"behaviour on an unreachable value; got {[r.name for r in off.disarmed]}"
    )
    on = audit_arming(_future_config(0.15), manifest=manifest)
    assert list(on.disarmed) == [], (
        "a positive threshold arms the row — CONFIG_THRESHOLD_GT_ZERO must be a real "
        "predicate over the value, not a constant"
    )
    assert list(off.deferred) == [] and list(on.deferred) == [], (
        "after the flip there is no deferred row left; `status` selects the list and it is "
        "DATA — no function may branch on the row's name"
    )


def test_the_mechanisms_are_real_predicates_in_both_directions() -> None:
    """`Mechanism` is the audit's only predicate authority (§8.3), so a constant here
    silently arms or disarms every row at once."""
    assert Mechanism.CONFIG_BOOL.is_armed(True) is True
    assert Mechanism.CONFIG_BOOL.is_armed(False) is False
    assert Mechanism.CONFIG_THRESHOLD_GT_ZERO.is_armed(0.15) is True
    assert Mechanism.CONFIG_THRESHOLD_GT_ZERO.is_armed(0.0) is False
    assert Mechanism.CONFIG_THRESHOLD_GT_ZERO.is_armed(-1.0) is False


# ── O-11 — SF-10: the burst override's central premise ────────────────────────────────
@pytest.mark.parametrize("name", CONFIG_PATHS)
def test_a_minted_config_survives_a_dump_revalidate_round_trip(name: str) -> None:
    """§5.3's mechanism is `load_config(p).model_dump()` → mutate ONE key →
    `RunConfig.model_validate(...)`, which is byte-for-byte the loader's own final step
    (`loader.py:39-44`). That is sound at HEAD **by accident**: a single future `alias`,
    `computed_field`, `exclude=` or `field_serializer` breaks it. This oracle is where that
    breakage lands, instead of inside the mint preflight on the training box.

    The leaf floor is the vacuity guard — two empty dicts compare equal just fine.
    """
    original = load_config(REPO_ROOT / "configs" / name)
    dumped = original.model_dump()

    def _leaves(node) -> int:
        if isinstance(node, dict):
            return sum(_leaves(value) for value in node.values())
        if isinstance(node, (list, tuple)):
            return sum(_leaves(item) for item in node)
        return 1

    assert _leaves(dumped) >= 100, (
        f"{name}: model_dump() produced {_leaves(dumped)} leaves — too few for the "
        "round-trip comparison to mean anything (measured at DESIGN: 194 per config)"
    )
    round_tripped = RunConfig.model_validate(dumped)
    assert round_tripped == original, (
        f"{name}: RunConfig.model_validate(cfg.model_dump()) != cfg — the burst override's "
        "central premise (§5.3) is broken; the override must not ship until it holds"
    )
    assert round_tripped.model_dump() == dumped, (
        f"{name}: the round trip is not dump-stable, so `override.booted_config_sha256` "
        "would not be reproducible"
    )
