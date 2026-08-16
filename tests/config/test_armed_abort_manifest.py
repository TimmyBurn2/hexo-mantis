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
- M10 `test_the_pinned_row_source_pin_is_tamper_evident` — R56's asymmetry: a pin that
      matches NOTHING is itself a hard failure, and so is a pin whose FILE is gone. Sole
      witness to the missing-pinned-file arm (`verify_source_pins`' vacuity direction).
      RENAMED at WPMINT DR-10 (R73): it was `test_the_deferred_row_source_pin_is_tamper_
      evident`, which R87's four hunks missed. The manifest carries ZERO deferred rows
      since Phase D's flip, so the old name named nothing; the subject was and still is
      "the manifest's pinned row", which is now the REQUIRED draw-rate row.
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
from mantis.config.loader import discover_configs, load_config
from mantis.config.schema import RunConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "ci_gates" / "preflight_mint.py"
MANIFEST_MODULE = REPO_ROOT / "src" / "mantis" / "config" / "armed_aborts.py"
_CONFIGS_DIR = REPO_ROOT / "configs"
# N4 (dispatcher-ownable backlog, F-P2B/N4): a flat `*.yaml` glob here is a SECOND answer to
# "what is a config" — blind to `configs/prod/run6.yaml`, which gate 7 and gate 12 both make
# legal (`discover_configs`, R71/R75, is the ONE discovery authority both gates consume).
# Relative-posix, not `.name`: a nested config's path must survive round-tripping back
# through `REPO_ROOT / "configs" / name` below.
CONFIG_PATHS = tuple(sorted(p.relative_to(_CONFIGS_DIR).as_posix() for p in discover_configs(_CONFIGS_DIR)))


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
    # FSTRING_MIDDLE exists only on 3.12+ (on the 3.11 floor f-strings tokenize as STRING,
    # so the skip set is complete there without it) — an unconditional attribute read is an
    # AttributeError on the pinned CI interpreter (WPCLEAN Phase LT).
    skip = {tokenize.COMMENT, tokenize.STRING}
    _fstring_middle = getattr(tokenize, "FSTRING_MIDDLE", None)
    if _fstring_middle is not None:
        skip.add(_fstring_middle)
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
                                   "N_pool_min": 50, "consec": 3}},
    )
    assert list(audit_arming(dev_armed).disarmed) == [], (
        "dev_example with BOTH armings flipped ON must PASS — otherwise the audit is keyed "
        "on the filename and assertion (c) is decoration"
    )


# ── M10 — R56 tamper-evidence ─────────────────────────────────────────────────────────
def test_the_pinned_row_source_pin_is_tamper_evident(tmp_path) -> None:
    """`source_pin` is `(repo-relative path, exact source text)` and the scan asserts the
    string is still THERE. R56's asymmetry (`silent_encoding_gate.py:338-344`): a pin that
    matches nothing is a HARD failure, not a quiet pass.

    R73 NAME-TRUTH, WPMINT DR-10. This test was `test_the_deferred_row_source_pin_is_tamper_
    evident` and its prose said it would go red "when Phase D deletes `draw_rate_threshold:
    float = 0.0` (R65)". Phase D landed: that literal is gone, the manifest holds ZERO
    `Status.DEFERRED` rows (asserted by
    `test_the_required_row_is_audited_against_a_REAL_RunConfig`), and R87's four re-prose
    hunks missed this one. The SUBJECT never moved — it is whatever row the manifest pins,
    which is now the REQUIRED draw-rate row whose pin binds `run.py`'s resolver threading —
    so the test is renamed and re-prosed rather than deleted or re-pointed.

    SCOPE, and why it is not O-D5's duplicate. This is the generic
    `verify_source_pins`-asymmetry oracle: it is the sole witness to the MISSING-FILE arm
    (`REVIEW_IMPL_P.md` RR-35, `FIX_ADJ13.md` V1). `test_the_required_row_keeps_a_source_
    pin_bound_to_the_construction_site` (O-D5) asserts WHICH row is pinned and WHAT the
    pinned text must contain; this one asserts only that the scan is asymmetric in all
    three directions. Neither subsumes the other.

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
        (row,), repo_root=tampered_root)] == [row.name], (
        f"deleting the pinned text from {rel} must report exactly the {row.name!r} row as "
        "broken — the pin is what makes an edit at the pinned site visible to gate 12"
    )

    absent_root = tmp_path / "absent"
    absent_root.mkdir()
    assert [broken.name for broken in TOOL.verify_source_pins(
        MANIFEST, repo_root=absent_root)] == [candidate.name for candidate in pinned], (
        "a pinned file that does not exist must be reported broken, never skipped — "
        "'nothing to scan' is how a tamper-evidence gate goes silently vacuous. EVERY pinned "
        "row must be reported here, not just the first: a scan that stopped at one would go "
        "half-vacuous the moment a second row was pinned (WPMINT K-B added one)"
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
    # WPMINT Phase K-B (call K-c) UPDATES the post-flip fact: the deferred list is no longer
    # empty. `grad_norm_hard_abort` is exactly the row R81 and R87 both predicted would arrive
    # ("CARD-COORD-KNOBS will feed it rows"), so the machinery kept alive for it now has a
    # SHIPPED subject and the rule below is asserted on the real manifest first. The synthetic
    # probe stays, because it is still the only way to drive a row whose pinned file is ABSENT.
    assert [row.name for row in _deferred()] == ["grad_norm_hard_abort"], (
        "the shipped manifest's deferred set is exactly the grad-norm row (WPMINT K-B): a "
        "live gate whose threshold nobody pre-registered, printed loudly and gating nothing. "
        f"A row appearing or vanishing here is a mint-visible change; got {_deferred()}"
    )
    for shipped in _deferred():
        rel, text = shipped.source_pin
        assert (REPO_ROOT / rel).is_file() and text in (REPO_ROOT / rel).read_text(), (
            f"deferred row {shipped.name!r} pins {rel!r}/{text!r}, which does not resolve in "
            "the real tree — a deferred row that is not tamper-evident rots into the status quo"
        )
    probe = ArmedAbort(
        name="_synthetic_deferred_probe", config_path="train.does_not_exist",
        mechanism=Mechanism.CONFIG_BOOL, status=Status.DEFERRED, exit_code=None,
        owner="CARD-COORD-KNOBS (R78)",
        source_pin=("src/mantis/config/armed_aborts.py", "class ArmedAbort"),
        note="synthetic subject for the deferred-row invariants; not a shipped row.",
    )
    deferred_rows = _deferred((*MANIFEST, probe))
    assert [row.name for row in deferred_rows] == ["grad_norm_hard_abort", probe.name], (
        "`_deferred` selects on `status` and nothing else — a selector that branched on a "
        "row's name or returned a constant is what this arm refuses. It must return BOTH the "
        "shipped deferred row (grad-norm, WPMINT K-B) and the synthetic one, in manifest "
        f"order; got {deferred_rows}"
    )
    for row in [candidate for candidate in deferred_rows if candidate.name == probe.name]:
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
    # WPMINT Phase K-B: the shipped deferred set is the grad-norm row and nothing else. The
    # DRAW-RATE row must not be among them — that is the fact this arm protects, and it is
    # stated directly now that "empty" has stopped being true.
    assert "draw_rate_collapse" not in [other.name for other in _deferred()], (
        "the flip is landed: the draw-rate row must not be deferred (a deferred row prints "
        f"and does not gate); got {_deferred()}"
    )
    row = ArmedAbort(
        name="draw_rate_collapse", config_path="train.draw_rate_abort.threshold",
        mechanism=Mechanism.CONFIG_THRESHOLD_GT_ZERO, status=Status.DEFERRED,
        exit_code=None, owner="CARD-COORD-KNOBS (R78)",
        source_pin=("src/mantis/run.py", "def compose_run"),
        note="synthetic pre-flip subject; the shipped row is REQUIRED since Phase D.",
    )
    assert row.name in [other.name for other in _deferred((*MANIFEST, row))], (
        "the DEFERRED machinery still works and still selects on status alone — that is what "
        "makes the flip below a DATA edit rather than a code change. (The shipped grad-norm "
        "row is deferred too since WPMINT K-B, so this asks for membership, not identity)"
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
        # WPMAIN RT-2/R132 adds `monitor.disk_guard` to the stub: the manifest gained a third
        # REQUIRED row whose arming surface is `monitor.disk_guard.fail_gb`, and a stub that
        # omits a REQUIRED row's surface raises `ArmingSurfaceMissingError` — correctly, that
        # is the phantom-input guard doing its job. The value is the minted 5.0 and it is
        # ARMED, so the disk-guard row never enters `disarmed` and this test's subject (the
        # draw-rate flip) is unchanged.
        return SimpleNamespace(
            monitor=SimpleNamespace(actor_lag_abort_enabled=True,
                                    disk_guard=SimpleNamespace(fail_gb=5.0)),
            train=SimpleNamespace(
                # WP12-R Phase O adds a FOURTH REQUIRED row whose arming surface is
                # `train.terminal_eval_enabled`; a stub that omits a REQUIRED row's surface
                # raises `ArmingSurfaceMissingError` — correctly, the phantom-input guard
                # doing its job. Minted `true` on all six committed configs, so the row is
                # ARMED, never enters `disarmed`, and this test's subject (the draw-rate
                # flip) is unchanged. Same maintenance RT-2/R132 did one row earlier.
                terminal_eval_enabled=True,
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
    assert ([r.name for r in off.deferred] == [r.name for r in on.deferred]
            == ["grad_norm_hard_abort"]), (
        "the draw-rate row's synthetic flip must not disturb the deferred list, which holds "
        "exactly the shipped grad-norm row (WPMINT K-B): `status` selects the list and it is "
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

    # WPMINT Phase K-B (call K-c) — the UPPER-bounded mechanism, real in BOTH of its two
    # inputs. `CONFIG_THRESHOLD_GT_ZERO` cannot judge `train.hard_gn_threshold`: its range is
    # genuinely unbounded above, so the shipped `1e9` reads ARMED while no finite gradient
    # norm reaches it. Every arm below moves ONE operand, so neither the value nor the
    # ceiling can be the constant this test exists to refuse.
    below = Mechanism.CONFIG_THRESHOLD_BELOW_CEILING
    assert below.is_armed(5.0, ceiling=10.0) is True
    assert below.is_armed(10.0, ceiling=10.0) is True, "the ceiling itself is IN range"
    assert below.is_armed(1e9, ceiling=10.0) is False, (
        "the shipped grad-norm threshold against the shipped monitor.alert_grad_norm_max — "
        "this False is the whole reason the mechanism exists"
    )
    assert below.is_armed(1e9, ceiling=1e10) is True, (
        "…and the SAME value must arm once the ceiling moves above it. Without this arm the "
        "predicate could ignore its ceiling and still pass every other line here"
    )
    assert below.is_armed(0.0, ceiling=10.0) is False
    assert below.is_armed(-1.0, ceiling=10.0) is False
    assert below.is_armed(5.0, ceiling=None) is False, (
        "a row with no usable ceiling must report DISARMED — an unjudgeable row fails toward "
        "visibility, never toward silence"
    )
    assert below.is_armed(float("inf"), ceiling=10.0) is False
    assert below.is_armed(5.0, ceiling=float("nan")) is False
    assert below.is_armed(True, ceiling=10.0) is False, "a bool is not a threshold"

    # …and the ceiling is DATA on the row, enforced in both directions.
    common = dict(name="probe", config_path="train.hard_gn_threshold",
                  status=Status.REQUIRED, exit_code=None, owner=None, source_pin=None,
                  note="oracle probe")
    with pytest.raises(ValueError, match="ceiling_path"):
        ArmedAbort(mechanism=Mechanism.CONFIG_THRESHOLD_BELOW_CEILING, **common)
    with pytest.raises(ValueError, match="ceiling_path"):
        ArmedAbort(mechanism=Mechanism.CONFIG_BOOL, ceiling_path="monitor.axis_warn", **common)
    ArmedAbort(mechanism=Mechanism.CONFIG_THRESHOLD_BELOW_CEILING,
               ceiling_path="monitor.alert_grad_norm_max", **common)


def test_the_grad_norm_row_reads_its_ceiling_off_the_real_config(smoke_run_config) -> None:
    """The DEFERRED grad-norm row (WPMINT Phase K-B, call K-c), audited against a REAL
    RunConfig in both directions.

    `Mechanism.is_armed` is a pure predicate; this is the other half — `audit_arming` must
    RESOLVE the row's `ceiling_path` on the config and hand it over, or the mechanism's second
    operand is a claim nothing feeds (LAW-07's phantom-input class). The row is DEFERRED, so
    it is flipped to REQUIRED in an in-memory copy to make it audit at all: that is the same
    `manifest=` seam O-7 uses, and it is also the exact edit that CLOSES the row, so this test
    is a rehearsal of the close.
    """
    row = [candidate for candidate in MANIFEST
           if candidate.name == "grad_norm_hard_abort"][0]
    assert row.status is Status.DEFERRED and row.exit_code is None, (
        "the row must stay DEFERRED with no invented exit code: flipping it REQUIRED would "
        "gate run5's mint on a threshold nobody pre-registered (R84's class)"
    )
    assert row.ceiling_path == "monitor.alert_grad_norm_max"

    def _required(manifest_row):
        return ArmedAbort(
            name=manifest_row.name, config_path=manifest_row.config_path,
            ceiling_path=manifest_row.ceiling_path, mechanism=manifest_row.mechanism,
            status=Status.REQUIRED, exit_code=manifest_row.exit_code, owner=None,
            source_pin=manifest_row.source_pin, note=manifest_row.note,
        )

    manifest = (_required(row),)
    shipped = load_config(REPO_ROOT / "configs" / "run5.yaml")
    assert [r.name for r in audit_arming(shipped, manifest=manifest).disarmed] == [row.name], (
        "as shipped (threshold 1e9 against alert_grad_norm_max 10.0) the gate is DISARMED — "
        "that is the finding the row exists to publish"
    )

    reachable = smoke_run_config("run5.yaml", train={"hard_gn_threshold": 5.0})
    assert list(audit_arming(reachable, manifest=manifest).disarmed) == [], (
        "a threshold at or below the warn line ARMS the row — the audit must read the CONFIG "
        "through both paths, not a constant"
    )
    raised = smoke_run_config("run5.yaml", monitor={"alert_grad_norm_max": 1e10})
    assert list(audit_arming(raised, manifest=manifest).disarmed) == [], (
        "and raising the CEILING alone must arm the SAME shipped threshold — the second "
        "operand really is resolved from `ceiling_path` and is not a literal"
    )


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


# ── ⊕ WP12-R Phase O / O-16 (R152/R84) — the terminal-eval-broken row ─────────────────
def test_the_terminal_eval_broken_row_is_required_and_imports_its_exit_code() -> None:
    """O-16. The rc R152 authors must be REGISTERED, not merely returned.

    R84's template is a registry entry plus a resolver, and the reason is one-authority: the
    number an operator reads off the process has to be the number the manifest publishes, or
    the manifest is documentation. `mantis.run` resolves the rc through
    `exit_code_for_abort`, which reads whatever `exit_code` the row carries — so a literal
    typed into the row and a literal typed at the launcher are the SAME defect one layer
    apart, and only the row's own provenance can stop it.

    `train.terminal_eval_enabled` is the arming surface and the row is REQUIRED, not
    DEFERRED, because nothing is owed: the field is a REQUIRED typed `bool` in the schema and
    is minted `true` on every committed config, so gate 12 is green the moment the row lands.
    What the row is FOR is the drift it makes loud — the day a production config is minted
    with the terminal eval off, gate 12 goes RED instead of the run quietly shipping with no
    terminal promotion decision (LAW-15: no promotion decision = deliverable incomplete).

    THE IMPORT-NOT-LITERAL CHECK IS AN AST CHECK, and that is not a style choice: CPython
    interns the small integers, so `row.exit_code is TERMINAL_EVAL_BROKEN_EXIT_CODE` is True
    even when the row types a bare `48` — an identity assertion CANNOT witness that mutation
    for any value in [-5, 256]. The only instrument that can is the source itself.

    MUTATION THAT REDS IT (M-O16): type `exit_code=48` as a literal in the row (the AST arm).
    MUTATION THAT REDS IT (M-O17): flip the row to `Status.DEFERRED` (the status arm)."""
    import ast

    from mantis.monitor.heartbeat import TERMINAL_EVAL_BROKEN_EXIT_CODE

    row = next((candidate for candidate in MANIFEST
                if candidate.name == "terminal_eval_broken"), None)
    assert row is not None, (
        "the armed-abort manifest authors no `terminal_eval_broken` row, so "
        "`exit_code_for_abort` answers None for the rule the composition root records and "
        "every broken terminal round exits 1 UnregisteredAbortExitError. Rows: "
        f"{[candidate.name for candidate in MANIFEST]}"
    )
    assert row.status is Status.REQUIRED and row.owner is None, (
        "a REQUIRED row carries no owner (an owner reads as already-excused debt); got "
        f"{row.status} / {row.owner!r}"
    )
    assert row.config_path == "train.terminal_eval_enabled", (
        "the arming surface is the NEARER condition — it gates the terminal round "
        f"specifically, where `eval_enabled` gates all eval; got {row.config_path!r}"
    )
    assert row.mechanism is Mechanism.CONFIG_BOOL, (
        f"a bool field is armed by `value is True`; got {row.mechanism}"
    )
    assert row.exit_code == TERMINAL_EVAL_BROKEN_EXIT_CODE == 48, (
        f"the row publishes the registered code; got {row.exit_code!r}"
    )
    assert row.source_pin is not None, (
        "R56 tamper-evidence: the row must pin the site that records its rule, or the row "
        "can go on claiming a fire path that was deleted"
    )
    rel, text = row.source_pin
    assert text in (REPO_ROOT / rel).read_text(encoding="utf-8"), (
        f"the pinned text {text!r} is gone from {rel} — the mechanism the rc depends on was "
        "deleted, renamed or reordered"
    )

    tree = ast.parse(MANIFEST_MODULE.read_text(encoding="utf-8"), filename=str(MANIFEST_MODULE))
    rows_in_source = [node for node in ast.walk(tree)
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                      and node.func.id == "ArmedAbort"]
    assert len(rows_in_source) == len(MANIFEST), (
        f"premise: every shipped row is constructed in {MANIFEST_MODULE.name}; the AST found "
        f"{len(rows_in_source)} constructions against {len(MANIFEST)} rows, so the census "
        "below would be reading a different set than the one under test"
    )
    typed_literals = [
        ast.unparse(keyword.value)
        for call in rows_in_source for keyword in call.keywords
        if keyword.arg == "exit_code" and isinstance(keyword.value, ast.Constant)
        and isinstance(keyword.value.value, int) and not isinstance(keyword.value.value, bool)
    ]
    assert typed_literals == [], (
        f"an armed-abort row types its exit code as a literal ({typed_literals}) instead of "
        "importing it from `mantis.monitor.heartbeat`. That is a second place the number is "
        "written: move the constant and the row goes on publishing the old one, with the "
        "resolver, the preflight parent and repo_design §11 all disagreeing about it"
    )


def test_the_terminal_eval_broken_row_is_armed_on_every_production_config() -> None:
    """O-16, the audit half — the row is not merely well-formed, it is ARMED where it counts.

    `audit_arming` resolves `train.terminal_eval_enabled` on the real production config and
    reports the row DISARMED if it is not `True`. Asserted here as the positive: gate 12 is
    green on run5 the day the row lands and NO armed value moves (R119's hard stop is
    untouched — 0.25 / 25000 / 50 are not on this axis at all)."""
    for name in PRODUCTION_CONFIGS:
        config = load_config(REPO_ROOT / name)
        audit = audit_arming(config)
        disarmed = [row.name for row in audit.disarmed]
        assert disarmed == [], (
            f"{name}: REQUIRED rows are disarmed: {disarmed}. Minting this config re-enables "
            "the failure each abort exists to catch"
        )
        armed = [row.name for row in audit.required]
        assert "terminal_eval_broken" in armed, (
            f"{name}: the terminal-eval row must be among the audited REQUIRED rows — a row "
            f"nobody audits is a registry entry, not a gate. Audited: {armed}"
        )
