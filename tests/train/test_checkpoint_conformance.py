# >300 justify (R8). Pre-existing gap, closed
# by WPMINT Phase W. This is ⊕⊕ Suite A, an oracle-first conformance suite whose every test is
# numbered against a PREREG row (`T-CK-*`) and was written and reviewed BEFORE the port existed.
# Splitting it renumbers nothing but breaks the 1:1 test->spec mapping REVIEW-design reads it
# through, and the suite's value is that ONE file answers "does this envelope conform" — the
# LAW-12 stamp/provenance/loader contract has one subject, not four.
"""⊕⊕ Suite A — checkpoint CONFORMANCE (WP10, 33 tests: T-CK-01 … T-CK-33).

Written oracle-first against repo_design §6 (the envelope-v2 contract) + the dispatcher
old-side captures (`wp/WP10/oldside/*`, pinned into `tests/fixtures/train/*.json`) BEFORE any
port code. The suite is RED until IMPL lands `mantis.train.checkpoints` (Slice 1) — importing
the not-yet-written module is the correct oracle-first state; IMPL turns it green.

Each test's docstring cites its `T-CK-*` id + the one-line PASS bar from `wp/WP10/PREREG.md`
so REVIEW-design can map test→spec 1:1. Tests assert on the DESIGN §c PUBLIC surfaces, never
private internals.

Slice note (see ORACLE_NOTES.md J1): the resume-precedence tests T-CK-14/15/17 read from
`mantis.train.orchestrator` and T-CK-18/19 need `mantis.train.trainer.core.Trainer` — both
Slice 2. Those five import their Slice-2 symbols LAZILY (inside the test) so the other 28 tests
can go green at Slice 1 (checkpoints only), matching the DESIGN IMPL-DAG gate for Suite A.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pytest
import torch

from mantis.config.loader import load_config
from mantis.config.schema import ARCH_SCOPED_KEYS
from mantis.encoding import EncodingRegistryError
from mantis.model import CnnArch, GnnArch, RepresentationMismatch  # noqa: F401 (arch types)

# ── Slice 1 surface under conformance (RED until IMPL writes train/checkpoints.py) ─────────
import mantis.train.checkpoints as checkpoints


# WP11-A schema extension: eval.gate/eval.ladder are now required fields (design §c.1).
def _make_eval_block() -> dict:
    return {
        "random_model_sims": 96, "sealbot_model_sims": 128, "kraken_model_sims": 128,
        "strix_model_sims": 128, "random_floor_games": 0, "worker_device": "cuda",
        "round_timeout_sec": 3600.0, "worker_kill_grace_sec": 10.0,
        "ply_cap_adjudication": None, "strength_floor": None,
        "gate": {
            "stride": 1, "screen_games": 80, "confirm_games": 128, "promotion_winrate": 0.55,
            "screen_confirm_lo": 0.44, "deploy_sims": 150, "opening_book": "book_v1_s20260625_p4",
            "bootstrap_resamples": 1000, "min_distinct_per_pair": 10, "seed_base": 20260625,
        },
        "ladder": {
            "rungs": [{"name": "sealbot_d5", "bot": "sealbot", "variant": "d5", "depth": 5,
                      "opponent_sims": None, "opening_book": "book_v1_s20260625_p4",
                      "deploy_matched": True, "games_max": 32}],
            "round_games": 64, "min_games_per_active_rung": 4, "graduation_wr_lower_ci": 0.75,
            "graduation_consec_rounds": 3, "activation_wr_lower_ci": 0.65,
            "calibration_every_k_rounds": 4, "calibration_games": 8,
            "bootstrap_resamples": 1000, "bootstrap_ci_level": 0.95,
            "bt_prior_games": 1.0, "bootstrap_seed": 1234,
        },
    }


from mantis.train.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    Checkpoint,  # noqa: F401 — the in-memory loaded-envelope view (public dataclass)
    CheckpointMetadata,  # noqa: F401
    CheckpointStampError,
    DeclaredEncodingMismatchError,
    apply_config_overrides_f1,
    checkpoint_filename,
    content_sha8,
    load_checkpoint,
    load_legacy_weights,
    resolve_lr_provenance,
    resume_trainer,
    save_checkpoint,
    strip_and_restamp,
)

KILLED_PREFIXES = ("cluster_pool.", "global_encoder.", "gpool_bias_branch.")

# The 18-key checkpoint-owned set, pinned LOCALLY (T-CK-15 mutation self-test: mutating the
# real constant makes the equality bite). Source: DESIGN §c.2 == old constant (#C3 verified).
_FROZEN_OWNED_LOCAL = frozenset({
    "encoding", "cluster_window_size", "cluster_threshold", "legal_move_radius", "board_size",
    "in_channels", "input_channels", "res_blocks", "filters", "se_reduction_ratio", "model",
    "total_steps", "scheduler_t_max", "eta_min", "min_lr", "lr", "weight_decay", "lr_schedule",
})


class _Evil:
    """A non-tensor picklable object — `weights_only=True` must refuse to unpickle it."""


# ── module helpers (operate on the on-disk payload; key names per repo_design §6) ──────────
def _save_full(tmp: Path, *, net, opt, scaler, sched, config, meta, step: int = 100,
               kind: str = "full", allow_quarantine: bool = False) -> Path:
    return save_checkpoint(
        model=net, optimizer=opt, scaler=scaler, scheduler=sched, step=step,
        config=config, metadata_kwargs=meta, checkpoint_dir=tmp, kind=kind,
        allow_quarantine=allow_quarantine,
    )


def _load_raw(path: Path) -> dict[str, Any]:
    """The on-disk v2 payload (all weights-only-safe types: dict/list/str/int/tensor)."""
    return torch.load(path, weights_only=True)


def _resave_rehashed(payload: dict[str, Any], checkpoint_dir: Path) -> Path:
    """Re-save a mutated payload to its correct `{run_id}_{step:08d}_{sha8}.ckpt` name (so a
    stale content-hash never masks the field under test). run_id/step read from metadata."""
    md = payload["metadata"]
    sha8 = content_sha8(payload)
    p = Path(checkpoint_dir) / checkpoint_filename(md["run_id"], md["step"], sha8)
    torch.save(payload, p)
    return p


def _overrides(result: Any) -> dict[str, Any]:
    """build_resume_config_overrides may return a bare overrides dict or a richer object with
    an `.overrides` field — accept either (return-shape flagged in ORACLE_NOTES J4)."""
    return dict(getattr(result, "overrides", result))


# ═══ Envelope v2 fields ════════════════════════════════════════════════════════════════════
def test_full_envelope_has_v2_schema_fields(tmp_path, tiny_net, optim_scaler_sched,
                                            valid_config, metadata_kwargs):
    """T-CK-01 — PASS iff the full payload has schema_version==2, kind=='full', model_state,
    optimizer/scaler/scheduler_state, config, metadata{encoding_name,run_id,step,commit_sha,
    created_utc,arch,corpus_sha256?}. Bites: dropping any required field / schema_version≠2."""
    opt, scaler, sched = optim_scaler_sched
    path = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                      config=valid_config, meta=metadata_kwargs, step=100, kind="full")
    ck = load_checkpoint(path)
    assert ck.schema_version == CHECKPOINT_SCHEMA_VERSION == 2
    assert ck.kind == "full"
    assert ck.model_state
    assert ck.optimizer_state is not None
    assert ck.scaler_state is not None
    assert ck.scheduler_state is not None
    assert isinstance(ck.config, dict) and ck.config
    md = ck.metadata
    assert md.encoding_name == "v6_live2_ls"
    assert md.run_id
    assert md.step == 100
    assert md.commit_sha  # present (may be "unknown" outside a git checkout) — never blocks
    assert md.created_utc
    assert isinstance(md.arch, (CnnArch, GnnArch))
    assert hasattr(md, "corpus_sha256")  # optional field exists on the dataclass


def test_weights_envelope_has_v2_schema_fields(tmp_path, tiny_net, optim_scaler_sched,
                                               valid_config, metadata_kwargs):
    """T-CK-02 — PASS iff a weights save is kind=='weights' with model_state + metadata and NO
    optimizer/scaler/scheduler_state. Bites: a weights save leaking optimizer state / missing kind."""
    opt, scaler, sched = optim_scaler_sched
    path = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                      config=valid_config, meta=metadata_kwargs, step=100, kind="weights")
    ck = load_checkpoint(path)
    assert ck.kind == "weights"
    assert ck.model_state
    assert ck.metadata is not None
    assert ck.optimizer_state is None
    assert ck.scaler_state is None
    assert ck.scheduler_state is None


def test_config_snapshot_schema_validated_on_write(tmp_path, tiny_net, optim_scaler_sched,
                                                   valid_config, invalid_config, metadata_kwargs):
    """T-CK-03 — PASS iff an invalid/incomplete config raises on write while a complete valid
    config saves. Bites: writing an unvalidated config snapshot."""
    opt, scaler, sched = optim_scaler_sched
    with pytest.raises(ValueError):  # pydantic ValidationError ⊂ ValueError
        _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                   config=invalid_config, meta=metadata_kwargs)
    good = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                      config=valid_config, meta=metadata_kwargs)
    assert good.exists()


def test_config_snapshot_schema_validated_on_read(tmp_path, tiny_net, optim_scaler_sched,
                                                  valid_config, metadata_kwargs):
    """T-CK-04 — PASS iff loading an envelope whose embedded config fails schema raises. Bites:
    a loader that skips config re-validation."""
    opt, scaler, sched = optim_scaler_sched
    path = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                      config=valid_config, meta=metadata_kwargs)
    payload = _load_raw(path)
    payload["config"]["schema_version"] = 999  # invalid: schema_version must be 1
    bad = _resave_rehashed(payload, tmp_path)
    with pytest.raises(ValueError):
        load_checkpoint(bad)


def test_metadata_encoding_name_required(tmp_path, tiny_net, optim_scaler_sched, valid_config,
                                         tiny_arch):
    """T-CK-05 — PASS iff a save whose encoding_name cannot be resolved raises (no metadata-
    omitted fallback). Bites: the old 'legacy caller → warn + omit metadata' path returning."""
    opt, scaler, sched = optim_scaler_sched
    meta_no_enc = {"run_id": "runa", "arch": tiny_arch}  # encoding_name MISSING
    with pytest.raises(CheckpointStampError):
        _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                   config=valid_config, meta=meta_no_enc)


# ═══ Filename (run-id + content-hash) ════════════════════════════════════════════════════════
def test_filename_carries_run_id_step_sha8(tmp_path, tiny_net, optim_scaler_sched, valid_config,
                                           mk_meta, tiny_arch):
    """T-CK-06 — PASS iff basename == {run_id}_{step:08d}_{sha8}.ckpt and sha8 is the first 8
    hex of the payload content hash. Bites: the old checkpoint_{step}.pt name / missing run_id
    or hash."""
    opt, scaler, sched = optim_scaler_sched
    meta = mk_meta(tiny_arch, run_id="runa")
    path = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                      config=valid_config, meta=meta, step=100)
    assert path.suffix == ".ckpt"
    run_id, step_str, sha8 = path.stem.rsplit("_", 2)
    assert run_id == "runa"
    assert step_str == "00000100"
    assert len(sha8) == 8 and all(c in "0123456789abcdef" for c in sha8)
    payload = _load_raw(path)
    assert content_sha8(payload) == sha8
    assert path.name == checkpoint_filename("runa", 100, sha8)


def test_cross_lineage_same_step_no_collision(tmp_path, tiny_net, optim_scaler_sched,
                                              valid_config, mk_meta, tiny_arch):
    """T-CK-07 — PASS iff two distinct run_id at the SAME step produce distinct filenames.
    Bites: a name that omits run_id (intricacy #4)."""
    opt, scaler, sched = optim_scaler_sched
    p1 = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                    config=valid_config, meta=mk_meta(tiny_arch, run_id="runa"), step=100)
    p2 = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                    config=valid_config, meta=mk_meta(tiny_arch, run_id="runb"), step=100)
    assert p1.name != p2.name
    assert p1.name.startswith("runa_00000100_")
    assert p2.name.startswith("runb_00000100_")


# ═══ Provenance re-verify at load ════════════════════════════════════════════════════════════
def test_load_reverifies_run_id_and_step(tmp_path, tiny_net, optim_scaler_sched, valid_config,
                                         mk_meta, tiny_arch):
    """T-CK-08 — PASS iff loading a file whose embedded run_id/step disagree with the filename
    raises a provenance error. Bites: a loader that trusts the filename and skips the re-check."""
    opt, scaler, sched = optim_scaler_sched
    path = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                      config=valid_config, meta=mk_meta(tiny_arch, run_id="runa"), step=100)
    sha8 = path.stem.rsplit("_", 2)[-1]
    wrong = path.with_name(checkpoint_filename("runa", 200, sha8))  # filename claims step 200
    path.rename(wrong)
    with pytest.raises(CheckpointStampError):
        load_checkpoint(wrong)


def test_tampered_payload_fails_content_hash(tmp_path, tiny_net, optim_scaler_sched, valid_config,
                                             metadata_kwargs):
    """T-CK-09 — PASS iff mutating one model_state element (sha8 unchanged in the name) makes
    load raise a content-hash mismatch. Bites: no content-hash re-verification at load."""
    opt, scaler, sched = optim_scaler_sched
    path = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                      config=valid_config, meta=metadata_kwargs)
    payload = _load_raw(path)
    k = next(iter(payload["model_state"]))
    t = payload["model_state"][k].clone()
    t.view(-1)[0] = t.view(-1)[0] + 1.0
    payload["model_state"][k] = t
    torch.save(payload, path)  # SAME filename → the name's sha8 is now stale
    with pytest.raises(CheckpointStampError):
        load_checkpoint(path)


# ═══ Stamp immutability ══════════════════════════════════════════════════════════════════════
def test_restamp_from_loaded_config_is_error(tmp_path, tiny_net, optim_scaler_sched, valid_config,
                                             metadata_kwargs):
    """T-CK-10 — PASS iff re-saving a loaded envelope while re-deriving metadata (new created_utc/
    commit_sha/run_id) FROM the loaded config raises. Bites: the self-perpetuating re-stamp
    (F-12/LAW-12). Realized as: supplying immutable stamp fields (created_utc/commit_sha) in
    metadata_kwargs is refused — stamps are minted ONCE by save (ORACLE_NOTES J5)."""
    opt, scaler, sched = optim_scaler_sched
    path = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                      config=valid_config, meta=metadata_kwargs)
    ck = load_checkpoint(path)
    restamp = {
        "encoding_name": ck.metadata.encoding_name,
        "run_id": ck.metadata.run_id,
        "arch": ck.metadata.arch,
        "created_utc": ck.metadata.created_utc,  # carrying an existing stamp = the F-12 bug
        "commit_sha": ck.metadata.commit_sha,
    }
    with pytest.raises(CheckpointStampError):
        _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                   config=ck.config, meta=restamp)


# ═══ Unstamped-write = failed save ═══════════════════════════════════════════════════════════
def test_unstamped_save_fails_loud_and_writes_nothing(tmp_path, tiny_net, optim_scaler_sched,
                                                      valid_config, tiny_arch):
    """T-CK-11 — PASS iff an unstampable save raises AND leaves no canonical file. Bites: a
    silent unstamped save."""
    opt, scaler, sched = optim_scaler_sched
    unstampable = {"run_id": "runa", "arch": tiny_arch}  # no encoding_name → cannot stamp
    with pytest.raises(CheckpointStampError):
        _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                   config=valid_config, meta=unstampable, allow_quarantine=False)
    assert list(tmp_path.glob("*.ckpt")) == []


def test_quarantine_path_when_run_must_survive(tmp_path, tiny_net, optim_scaler_sched, valid_config,
                                               tiny_arch, monkeypatch):
    """T-CK-12 — PASS iff, with the survive-run flag, an unstampable save writes <path>.quarantine
    + increments the QUARANTINE counter and NOT the persist-fatal one, NEVER a canonical name.
    Bites: a canonical unstamped artifact, and (post R-QUARANTINE-COUNTER, WPCLEAN Phase RES)
    a survivable quarantine leaking into the watchdog's `> 0` persist-fatal rule — the exact
    conflation the debt row recorded: the survive-run clause used to feed rc 43.

    Both counters are process-wide module GLOBALS incremented via `global … += 1`, which no
    assertion can undo. The monkeypatch pins them to 0 for the test AND RESTORES the pre-test
    values at teardown, so a leak cannot reach another suite (WP13-A REVIEW-impl F-2)."""
    monkeypatch.setattr(checkpoints, "persist_errors_total", 0)
    monkeypatch.setattr(checkpoints, "quarantine_writes_total", 0)
    opt, scaler, sched = optim_scaler_sched
    unstampable = {"run_id": "runa", "arch": tiny_arch}
    p = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                   config=valid_config, meta=unstampable, allow_quarantine=True)
    assert str(p).endswith(".quarantine")
    assert p.exists()
    assert list(tmp_path.glob("*.ckpt")) == []
    assert checkpoints.quarantine_writes_total == 1
    assert checkpoints.persist_errors_total == 0, (
        "a survivable quarantine fed the persist-FATAL counter — the watchdog would abort "
        "(rc 43) on the run this clause exists to save (R-QUARANTINE-COUNTER)"
    )


# ═══ weights_only on every load surface ══════════════════════════════════════════════════════
def test_every_load_surface_uses_weights_only_true(tmp_path, tiny_net, optim_scaler_sched,
                                                   valid_config, metadata_kwargs):
    """T-CK-13 — PASS iff (i) a checkpoint carrying a non-tensor picklable object fails to load
    and (ii) a source census finds ZERO `weights_only=False`. Bites: any weights_only=False surface."""
    opt, scaler, sched = optim_scaler_sched
    path = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                      config=valid_config, meta=metadata_kwargs)
    # (i) inject a non-tensor object → weights_only=True must reject it during torch.load.
    payload = _load_raw(path)
    payload["model_state"]["__evil__"] = _Evil()
    torch.save(payload, path)
    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(path)
    # (ii) source census — checkpoints (Slice 1) always; anchor (Slice 3) when present.
    import importlib.util
    src = Path(checkpoints.__file__).read_text()
    assert "weights_only=False" not in src
    anchor_spec = importlib.util.find_spec("mantis.train.anchor")
    if anchor_spec is not None and anchor_spec.origin:
        assert "weights_only=False" not in Path(anchor_spec.origin).read_text()


# ═══ Resume precedence + frozen key set ══════════════════════════════════════════════════════
def test_launch_config_wins_except_frozen_keys():
    """T-CK-14 — PASS iff a non-frozen override is applied while a frozen key (encoding/arch/
    optim/sched) defers to the checkpoint. Bites: the pre-E0 precedence inversion."""
    from mantis.train.orchestrator import build_resume_config_overrides  # Slice 2 (lazy)
    baked = {"aux_chain_weight": 0.5, "lr": 0.001, "encoding": "v6_live2_ls"}
    launch = {"aux_chain_weight": 0.2, "lr": 0.002, "encoding": "v6"}
    ov = _overrides(build_resume_config_overrides(baked, launch))
    assert ov.get("aux_chain_weight") == 0.2  # non-frozen → launch wins
    assert "lr" not in ov                       # frozen (checkpoint-owned) → excluded
    assert "encoding" not in ov                 # frozen → excluded


def test_frozen_key_set_is_the_pinned_constant(resume_goldens):
    """T-CK-15 — PASS iff RESUME_CHECKPOINT_OWNED_KEYS equals the exact 18-key set; mutating the
    constant bites. Bites: a silent change to the checkpoint-owned set."""
    from mantis.train.orchestrator import RESUME_CHECKPOINT_OWNED_KEYS  # Slice 2 (lazy)
    golden = resume_goldens["T-CK-15_frozen_key_set"]
    assert set(RESUME_CHECKPOINT_OWNED_KEYS) == set(golden["sorted_keys"])
    assert len(RESUME_CHECKPOINT_OWNED_KEYS) == golden["count"] == 18
    assert frozenset(RESUME_CHECKPOINT_OWNED_KEYS) == _FROZEN_OWNED_LOCAL


def test_declared_key_wins_base_inherited_defers(resume_goldens, spy_sink):
    """T-CK-16 — PASS iff a declared key wins over baked, a base-inherited key defers to baked
    (+ warns on differ), and a declared null travels. Bites: the F1(A) defer regression."""
    g = resume_goldens["T-CK-16_declared_wins_base_defers"]
    inp, exp = g["inputs"], g["expected_output"]
    resolved, deferred = apply_config_overrides_f1(
        dict(inp["baked_config"]), dict(inp["config_overrides"]),
        set(inp["declared_keys"]), sink=spy_sink,
    )
    assert resolved == exp["resolved_config"]
    assert set(deferred) == set(exp["deferred_keys"])
    warns = spy_sink.named(exp["warning_event"])
    assert warns, f"expected a {exp['warning_event']} event on the sink"
    assert warns[-1]["knob"] == exp["warning_fields"]["knob"]
    assert warns[-1]["base_default"] == exp["warning_fields"]["base_default"]
    assert warns[-1]["checkpoint_baked"] == exp["warning_fields"]["checkpoint_baked"]


# ═══ Scheduler / resume semantics ════════════════════════════════════════════════════════════
def test_scheduler_horizon_gate(resume_goldens):
    """T-CK-17 — PASS iff without the flag the horizon keys stay OWNED (excluded from overrides)
    and with it total_steps/scheduler_t_max re-enter. Bites: a silent scheduler re-horizon."""
    from mantis.train.orchestrator import build_resume_config_overrides  # Slice 2 (lazy)
    g = resume_goldens["T-CK-17_scheduler_horizon_gate"]
    baked, launch = g["inputs"]["baked_config_A"], g["inputs"]["launch_variant_B"]
    exp = g["expected_output"]
    ov_off = _overrides(build_resume_config_overrides(dict(baked), dict(launch),
                                                      override_scheduler_horizon=False))
    assert ("total_steps" in ov_off) is exp["override_scheduler_horizon_FALSE"]["overrides_contains_total_steps"]
    assert ("scheduler_t_max" in ov_off) is exp["override_scheduler_horizon_FALSE"]["overrides_contains_scheduler_t_max"]
    ov_on = _overrides(build_resume_config_overrides(dict(baked), dict(launch),
                                                     override_scheduler_horizon=True))
    assert ov_on.get("total_steps") == exp["override_scheduler_horizon_TRUE"]["overrides_total_steps"]
    assert ov_on.get("scheduler_t_max") == exp["override_scheduler_horizon_TRUE"]["overrides_scheduler_t_max"]


def test_missing_scheduler_state_requires_allow_fresh(tmp_path, tiny_net, optim_scaler_sched,
                                                      valid_config, metadata_kwargs, spy_sink,
                                                      resume_goldens):
    """T-CK-18 — PASS iff a full resume with scheduler_state None raises unless allow_fresh_scheduler
    (then it warns). Bites: a silent fresh-scheduler start. (Slice 2: needs Trainer + resume_trainer.)"""
    from mantis.train.trainer.core import Trainer  # Slice 2 (lazy)
    opt, scaler, _sched = optim_scaler_sched
    path = save_checkpoint(model=tiny_net, optimizer=opt, scaler=scaler, scheduler=None,
                           step=750, config=valid_config, metadata_kwargs=metadata_kwargs,
                           checkpoint_dir=tmp_path, kind="full")
    g = resume_goldens["T-CK-18_missing_scheduler_state"]["expected_output"]
    with pytest.raises(ValueError):
        resume_trainer(Trainer, path, fallback_config=valid_config)
    tr = resume_trainer(Trainer, path, fallback_config=valid_config,
                        config_overrides={"allow_fresh_scheduler": True}, sink=spy_sink)
    assert tr is not None
    assert spy_sink.has(g["allow_fresh_scheduler_TRUE"]["warning_event"])


def test_full_resume_restores_optimizer_scaler_step(tmp_path, tiny_net, optim_scaler_sched,
                                                    valid_config, metadata_kwargs, resume_goldens):
    """T-CK-19 — PASS iff a full resume restores optimizer(param_groups==2)/scaler/step==750 and a
    weights resume gets a fresh optimizer + step==500 (promoted-anchor recovery). Bites: dropping
    optimizer/scaler restore or losing the promoted-anchor step. (Slice 2: needs Trainer.)"""
    from mantis.train.trainer.core import Trainer  # Slice 2 (lazy)
    exp = resume_goldens["T-CK-19_full_vs_weights_restore"]["expected_output"]
    opt, scaler, sched = optim_scaler_sched
    full_path = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                           config=valid_config, meta=metadata_kwargs, step=750, kind="full")
    weights_path = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                              config=valid_config, meta=metadata_kwargs, step=500, kind="weights")
    tr_full = resume_trainer(Trainer, full_path, fallback_config=valid_config)
    assert tr_full.loaded_from_full_checkpoint is exp["full_resume"]["loaded_from_full_checkpoint"]
    assert tr_full.step == exp["full_resume"]["resumed_step"]
    assert len(tr_full.optimizer.param_groups) == exp["full_resume"]["optimizer_param_groups_restored"]
    assert tr_full.scaler is not None
    tr_w = resume_trainer(Trainer, weights_path, fallback_config=valid_config)
    assert tr_w.loaded_from_full_checkpoint is exp["weights_only_resume"]["loaded_from_full_checkpoint"]
    assert tr_w.step == exp["weights_only_resume"]["resumed_step_from_wrapper"]


def test_declared_lr_ignored_on_full_resume_is_loud(resume_goldens):
    """T-CK-20 — PASS iff a declared lr differing from the baked lr on a full resume is IGNORED
    (lr is resume-state-owned) with a loud warning. Bites: silent lr override / silent lr drop.
    Unit-level via resolve_lr_provenance; the end-to-end warning event
    `lr_declared_override_ignored_on_full_resume` is exercised by O-F1E0."""
    g = resume_goldens["T-CK-20_lr_resume_owned"]["expected_output"]
    ign = g["resolve_lr_provenance_override_ignored_case"]
    prov = resolve_lr_provenance(declared=ign["declared"], baked=ign["baked"],
                                 effective=ign["effective"])
    assert prov.override_ignored is True
    norm = resolve_lr_provenance(declared=0.001, baked=0.001, effective=0.001)
    assert norm.override_ignored is g["resolve_lr_provenance_normal_case_declared_eq_baked"]["override_ignored"]


# ═══ Weights-strip wire-signature gate ═══════════════════════════════════════════════════════
def test_weights_strip_requires_wire_signature_equality(tmp_path, tiny_net, optim_scaler_sched,
                                                        valid_config, metadata_kwargs):
    """T-CK-21 — PASS iff the weights-strip + re-stamp succeeds only on wire-signature equality;
    a mismatch raises. Bites: an encoding change with no wire-signature check. (strip_and_restamp
    is the inferred name for the DESIGN §6 'weights-only strip' path — ORACLE_NOTES J6.)"""
    opt, scaler, sched = optim_scaler_sched
    src = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                     config=valid_config, meta=metadata_kwargs)  # encoding v6_live2_ls
    same = strip_and_restamp(src, new_encoding="v6_live2_ls", run_id="runc",
                             checkpoint_dir=tmp_path)  # equal wire sig → OK
    assert Path(same).exists()
    with pytest.raises((CheckpointStampError, RepresentationMismatch, ValueError)):
        strip_and_restamp(src, new_encoding="v6", run_id="rund",
                          checkpoint_dir=tmp_path)  # v6 (8pl) ≠ v6_live2_ls (4pl) wire sig


# ═══ O3b — reject killed-branch prefixes ═════════════════════════════════════════════════════
def _save_bare(state: dict, path: Path) -> None:
    torch.save(state, path)


def _inject(state: dict, prefix: str) -> dict:
    dirty = dict(state)
    dirty[f"{prefix}some.weight"] = torch.zeros(1)
    return dirty


def _forge_v2_with_killed_prefix(tmp_path, *, net, opt, scaler, sched, config, meta,
                                 prefix: str) -> Path:
    """Build a REAL stamped v2 envelope, then hand-forge a killed prefix into its model_state and
    re-name to a valid {run_id}_{step}_{sha8}.ckpt (so the provenance/content-hash checks pass and
    the O3b reject — not a hash mismatch — is what must fire). A stamped v2 can STRUCTURALLY carry a
    killed key, so the mint-path 'build_net can't emit one' argument alone is insufficient."""
    good = _save_full(tmp_path, net=net, opt=opt, scaler=scaler, sched=sched, config=config,
                      meta=meta)
    payload = _load_raw(good)
    payload["model_state"][f"{prefix}some.weight"] = torch.zeros(1)
    return _resave_rehashed(payload, tmp_path)


def test_reject_cluster_pool_prefix(tmp_path, full_ls_state, tiny_net, optim_scaler_sched,
                                    valid_config, metadata_kwargs):
    """T-CK-22 — PASS iff a synthetic cluster_pool. key makes BOTH loader surfaces raise
    RepresentationMismatch (no PMA pool built): load_legacy_weights (bare) AND load_checkpoint
    (hand-forged v2). Bites: resurrecting _build_min_max_model's PMA sniff-reconstruct (F-04)."""
    opt, scaler, sched = optim_scaler_sched
    p = tmp_path / "dirty_cluster.pt"
    _save_bare(_inject(full_ls_state, "cluster_pool."), p)
    with pytest.raises(RepresentationMismatch):
        load_legacy_weights(p, declared_encoding="v6_live2_ls")
    forged = _forge_v2_with_killed_prefix(tmp_path, net=tiny_net, opt=opt, scaler=scaler,
                                          sched=sched, config=valid_config, meta=metadata_kwargs,
                                          prefix="cluster_pool.")
    with pytest.raises(RepresentationMismatch):
        load_checkpoint(forged)


def test_reject_global_encoder_prefix(tmp_path, full_ls_state, tiny_net, optim_scaler_sched,
                                      valid_config, metadata_kwargs):
    """T-CK-23 — PASS iff a global_encoder. key raises on BOTH surfaces (bare load_legacy_weights
    AND hand-forged v2 load_checkpoint). Bites: pma_global reconstruction."""
    opt, scaler, sched = optim_scaler_sched
    p = tmp_path / "dirty_global.pt"
    _save_bare(_inject(full_ls_state, "global_encoder."), p)
    with pytest.raises(RepresentationMismatch):
        load_legacy_weights(p, declared_encoding="v6_live2_ls")
    forged = _forge_v2_with_killed_prefix(tmp_path, net=tiny_net, opt=opt, scaler=scaler,
                                          sched=sched, config=valid_config, meta=metadata_kwargs,
                                          prefix="global_encoder.")
    with pytest.raises(RepresentationMismatch):
        load_checkpoint(forged)


def test_reject_gpool_bias_branch_prefix(tmp_path, full_ls_state, tiny_net, optim_scaler_sched,
                                         valid_config, metadata_kwargs):
    """T-CK-24 — PASS iff a gpool_bias_branch. key raises on BOTH surfaces (bare load_legacy_weights
    AND hand-forged v2 load_checkpoint). Bites: gpool-bias reconstruction (F-05)."""
    opt, scaler, sched = optim_scaler_sched
    p = tmp_path / "dirty_gpool.pt"
    _save_bare(_inject(full_ls_state, "gpool_bias_branch."), p)
    with pytest.raises(RepresentationMismatch):
        load_legacy_weights(p, declared_encoding="v6_live2_ls")
    forged = _forge_v2_with_killed_prefix(tmp_path, net=tiny_net, opt=opt, scaler=scaler,
                                          sched=sched, config=valid_config, meta=metadata_kwargs,
                                          prefix="gpool_bias_branch.")
    with pytest.raises(RepresentationMismatch):
        load_checkpoint(forged)


def test_clean_v6_live2_anchor_loads(tmp_path, full_ls_net, full_ls_state, anchor_key_set):
    """T-CK-25 — PASS iff the real bootstrap_model_v6_live2.pt key set (147, 0 killed) ⊆ the
    stripped build_net(arch_from_spec('v6_live2_ls')) key set and loads clean. Bites: a false-
    positive reject on a clean promoted anchor. (Fixture is committed v6_live2.txt — J2.)"""
    constructed = set(full_ls_net.state_dict().keys())
    assert len(anchor_key_set) == 147
    assert anchor_key_set <= constructed
    assert not any(k.startswith(KILLED_PREFIXES) for k in anchor_key_set)
    p = tmp_path / "clean_anchor.pt"
    _save_bare(full_ls_state, p)
    ck = load_legacy_weights(p, declared_encoding="v6_live2_ls")  # no RepresentationMismatch
    assert ck.model_state


def test_killed_prefix_reject_mutation_selftest(tmp_path, full_ls_state):
    """T-CK-26 (LAW-07) — PASS iff a clean state dict loads AND injecting a killed prefix makes
    the loader reject. Bites: a guard wired but never firing."""
    clean = tmp_path / "clean.pt"
    _save_bare(dict(full_ls_state), clean)
    assert load_legacy_weights(clean, declared_encoding="v6_live2_ls").model_state  # clean loads
    dirty = tmp_path / "dirty.pt"
    _save_bare(_inject(full_ls_state, "cluster_pool."), dirty)
    with pytest.raises(RepresentationMismatch):  # mutation → reject fires
        load_legacy_weights(dirty, declared_encoding="v6_live2_ls")


# ═══ Declared-encoding-assert vs decode-override ═════════════════════════════════════════════
def test_declared_encoding_mismatch_raises(tmp_path, tiny_net, optim_scaler_sched, valid_config,
                                           metadata_kwargs):
    """T-CK-27 — PASS iff declared_encoding disagreeing with the stamp raises
    DeclaredEncodingMismatchError naming both. Bites: a stamp silently overriding a declared name."""
    opt, scaler, sched = optim_scaler_sched
    path = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                      config=valid_config, meta=metadata_kwargs)  # stamp = v6_live2_ls
    with pytest.raises(DeclaredEncodingMismatchError):
        load_checkpoint(path, declared_encoding="v6")


def test_decode_override_wins_and_logs_never_raises(tmp_path, tiny_net, optim_scaler_sched,
                                                    valid_config, metadata_kwargs, caplog):
    """T-CK-28 — PASS iff decode_override is authoritative + logs `encoding_decode_override` loudly
    on disagreement but NEVER raises. Bites: override raising / overriding silently."""
    import logging
    opt, scaler, sched = optim_scaler_sched
    path = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                      config=valid_config, meta=metadata_kwargs)  # stamp = v6_live2_ls
    with caplog.at_level(logging.INFO):  # floor at INFO so a WARNING-or-INFO notice is captured
        ck = load_checkpoint(path, decode_override="v6")  # disagrees with stamp → no raise
    assert ck is not None
    assert "encoding_decode_override" in caplog.text


def test_declared_and_override_together_error(tmp_path, tiny_net, optim_scaler_sched, valid_config,
                                              metadata_kwargs):
    """T-CK-29 — PASS iff passing both declared_encoding and decode_override raises ValueError
    (mutually exclusive). Bites: allowing both."""
    opt, scaler, sched = optim_scaler_sched
    path = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                      config=valid_config, meta=metadata_kwargs)
    with pytest.raises(ValueError):
        load_checkpoint(path, declared_encoding="v6_live2_ls", decode_override="v6")


def test_stamp_sources_disagree_raises(tmp_path, tiny_net, optim_scaler_sched, valid_config,
                                       metadata_kwargs):
    """T-CK-30 — PASS iff a checkpoint whose metadata.encoding_name and config.encoding resolve to
    DIFFERENT names raises 'stamp sources disagree'. Bites: silently picking one source."""
    opt, scaler, sched = optim_scaler_sched
    path = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                      config=valid_config, meta=metadata_kwargs)  # both v6_live2_ls
    payload = _load_raw(path)
    payload["config"]["identity"]["encoding"] = "v6"  # config now says v6, metadata says v6_live2_ls
    bad = _resave_rehashed(payload, tmp_path)
    with pytest.raises(CheckpointStampError):
        load_checkpoint(bad)


# ═══ Legacy read (anchor import path) — the THREE real old shapes ════════════════════════════
def test_reads_full_v1_envelope_via_field_map(tmp_path, full_ls_net, full_ls_state, legacy_shapes):
    """T-CK-31 — PASS iff a full old envelope reads via the old→v2 field map (training_date→
    created_utc, model_architecture/variant→arch, train_config_path dropped, config re-validated)
    → resume-capable load. Bites: refusing a real pre-v2 full checkpoint / mis-mapping a field.

    The captured encoding 'v6_live2' is UNREGISTERED in the new repo (J3) → the synthetic full-v1
    envelope uses the registered 'v6_live2_ls' for arch-resolvability while pinning the captured
    field-map scalars (training_date value asserted verbatim)."""
    fv1 = legacy_shapes["full_v1_envelope"]
    md = fv1["metadata"]
    from mantis.model import arch_from_spec_and_config
    from mantis.encoding import lookup
    opt = torch.optim.AdamW(full_ls_net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000000, eta_min=0.0005)
    valid_config = {
        "schema_version": 1, "run_id": "run5", "seed": 20260718,
        "eval_enabled": True,
        # RECAL-PREP (R308(g)(i)): a REQUIRED top-level leaf. `null` is R119's
        # placeholder — refused at boot on a cuda process, valued only by the
        # re-calibration sitting under R282(b).
        "allocator_posture": None,
        "identity": {"encoding": "v6_live2_ls", "representation": "grid"},
        "eval": _make_eval_block(),
        # WPMINT Phase K-A stage 0: DERIVED from a MINTED config, not a twelfth restatement
        # of the complete `train:` block (measured byte-identical to the census it replaces).
        "train": load_config(
            Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml"
        ).train.model_dump(),
        "selfplay": {
            "n_workers": 1, "leaf_batch_size": 8, "max_game_moves": 128,
            "inference_pool_size": None, "completed_q_values": False, "c_visit": 50.0,
            "c_scale": 1.0, "gumbel_mcts": False, "gumbel_m": 16, "gumbel_explore_moves": 10,
            "results_queue_cap": 10_000, "random_opening_plies": 0, "rotation_enabled": True,
            "forced_win_policy_enabled": False, "forced_win_policy_depth": 2,
            "forced_win_policy_weight": 1.0, "solver_enabled": False, "solver_depth": 16,
            "solver_node_budget": 50_000, "solver_neighbor_dist": 2, "solver_visit_weight": 0.3,
            "seed_fraction": 0.0, "seed_corpus_path": None, "log_investigation_metrics": True,
            "instrumentation_enabled": False,
            "mcts": {"n_simulations": 50, "c_puct": 1.5, "fpu_reduction": 0.25,
                     "quiescence_enabled": True, "quiescence_blend_2": 0.3,
                     "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25,
                     "dirichlet_enabled": True},
            "playout_cap": {"fast_sims": 50, "fast_prob": 0.0, "standard_sims": 0,
                            "full_search_prob": 0.0, "n_sims_quick": 0, "n_sims_full": 0,
                            "zoi_enabled": False, "zoi_lookback": 16, "zoi_margin": 5,
                            "temperature_threshold_compound_moves": 0, "temp_min": 0.5},
        },
        "inference": {
            "inference_batch_size": 64, "inference_max_wait_ms": 10, "trace_inference": True,
            "compile_inference": False, "compile_inference_mode": "default",
            "compile_inference_dynamic": True, "perf_timing": False, "perf_sync_cuda": False,
            # (`fused_graph_caps` is ARCH-SCOPED to graph and this fixture is a GRID envelope,
            # so it is absent rather than non-binding — R322(d). The drop below removes it and
            # `train.microbatch_caps` from the dev-template blocks this payload borrows.)
        },
        "monitor": {
            # R242 (ADJ-D12): the ARMING cadence, schema-only and required.
            "gate_interval": 1000,
            "alert_entropy_min": 1.0, "collapse_threshold_nats": 1.5, "alert_grad_norm_max": 10.0,
            "alert_loss_increase_window": 3, "wr_hard_abort_enabled": False,
            "wr_rolling_consecutive_evals": 2, "wr_rolling_threshold": 0.10,
            "wr_rolling_min_step": 20000, "wr_collapse_from_peak_ratio": 0.5,
            "wr_collapse_min_step": 25000, "wr_collapse_consecutive_evals": 3,
            "wr_early_death_threshold": 0.05, "wr_early_death_min_step": 15000,
            "axis_warn": 0.45, "axis_alert": 0.50,
            "heartbeat_deadline_train_step_sec": 1800.0,
            "heartbeat_deadline_inference_dispatch_sec": 1800.0,
            "heartbeat_deadline_selfplay_drain_sec": 1800.0,
            "heartbeat_deadline_eval_round_sec": 1800.0,
            "heartbeat_poll_interval_sec": 5.0, "heartbeat_file_interval_sec": 15.0,
            "heartbeat_close_out_deadline_sec": 14400.0, "heartbeat_fire_effect_timeout_sec": 30.0,
            "supervisor_stale_after_sec": 900.0, "supervisor_poll_interval_sec": 30.0,
            "supervisor_kill_grace_sec": 30.0, "supervisor_max_relaunches": 5,
            "actor_lag_threshold_steps": 100, "actor_lag_abort_enabled": False,
            "drain": {
                "final_eval_drain_timeout_sec": 900.0, "eval_final_drain_safety_factor": 3.0,
                "eval_final_drain_hard_cap_sec": 14400.0, "terminal_eval_hard_cap_sec": 14400.0,
            },
                "disk_guard": {"interval_sec": 60.0, "warn_gb": 10.0, "fail_gb": 5.0},
        },
    }
    # R322(d): this is a GRID envelope, and the `train:` block above is borrowed from a GRAPH
    # config's dump, so it arrives carrying `microbatch_caps`. Both arch-scoped blocks are
    # dropped through `ARCH_SCOPED_KEYS` — the schema's own partition — rather than by name,
    # so a third scoped block needs no edit here.
    for _key in ARCH_SCOPED_KEYS:
        if valid_config["identity"]["representation"] != _key.arch:
            valid_config[_key.section].pop(_key.field, None)
    payload = {  # the real FULL-v1 top-level shape (7 keys) + captured metadata scalars
        "step": fv1["step"],
        "model_state": full_ls_state,
        "optimizer_state": opt.state_dict(),
        "scaler_state": torch.amp.GradScaler("cpu", enabled=True).state_dict(),
        "scheduler_state": sched.state_dict(),
        "config": valid_config,
        "metadata": {
            "encoding_name": "v6_live2_ls",             # registered (arch-resolvable); verbatim-passthrough
            "commit_sha": md["commit_sha"],
            "training_date": md["training_date"],       # → created_utc (rename, verbatim)
            "train_config_path": None,                  # → DROPPED
            "corpus_sha256": None,
            "model_architecture": md["model_architecture"],
            "model_variant": None,
            "schema_version": 1,                        # v1 → legacy read path
        },
    }
    legacy_path = tmp_path / "checkpoint_00272357.pt"  # NOT a v2 {run_id}_{step}_{sha8}.ckpt name
    torch.save(payload, legacy_path)
    ck = load_legacy_weights(legacy_path)
    assert ck.kind == "full"                            # optimizer/scaler/scheduler present → resume-capable
    assert ck.optimizer_state is not None
    assert ck.scheduler_state is not None
    assert ck.metadata.encoding_name == "v6_live2_ls"   # verbatim
    assert ck.metadata.commit_sha == md["commit_sha"]   # verbatim
    assert ck.metadata.created_utc == md["training_date"]  # training_date → created_utc, VERBATIM
    assert ck.metadata.corpus_sha256 is None
    assert not hasattr(ck.metadata, "train_config_path")   # dropped (not a v2 metadata field)
    assert not ck.metadata.run_id                        # SYNTHESIZED-NEVER on a legacy read
    assert isinstance(ck.metadata.arch, (CnnArch, GnnArch))  # resolved, not sniffed


def test_reads_bare_state_dict_anchor_no_fake_provenance(tmp_path, full_ls_state, legacy_shapes):
    """T-CK-32 — PASS iff a BARE state_dict anchor loads via load_legacy_weights (arch from the
    declared/registry encoding), kind='weights', NO synthetic run_id/hash/created_utc. Bites:
    minting fake v2 provenance for a legacy bare anchor (LAW-12) / refusing a bare anchor outright."""
    assert legacy_shapes["bare_state_dict"]["top_level_is_envelope"] is False
    bare = tmp_path / "bootstrap_model_v6_live2.pt"
    torch.save(full_ls_state, bare)  # the whole payload IS the state dict — no wrapper
    ck = load_legacy_weights(bare, declared_encoding="v6_live2_ls")
    assert ck.kind == "weights"
    assert ck.model_state
    assert ck.metadata.encoding_name == "v6_live2_ls"   # from declared, not embedded
    assert not ck.metadata.run_id                        # NO synthetic run_id
    assert not getattr(ck.metadata, "created_utc", "")   # NO synthetic created_utc
    assert ck.optimizer_state is None                    # weights-only


def test_bare_anchor_to_v2_requires_explicit_strip(tmp_path, full_ls_state):
    """T-CK-33 — PASS iff upgrading a legacy bare anchor to a stamped v2 envelope goes ONLY through
    the wire-signature-gated weights-only strip + re-stamp (stamped once from declared encoding +
    arch); an auto-restamp on read raises. Bites: an auto-restamp on read."""
    bare = tmp_path / "bootstrap_model_v6_live2.pt"
    torch.save(full_ls_state, bare)
    # sanctioned upgrade: strip + re-stamp → a proper v2 envelope with a FRESH single stamp.
    v2_path = strip_and_restamp(bare, new_encoding="v6_live2_ls", run_id="runx",
                                checkpoint_dir=tmp_path, declared_encoding="v6_live2_ls")
    ck = load_checkpoint(v2_path)
    assert ck.metadata.run_id == "runx"
    assert ck.metadata.created_utc  # stamped ONCE at strip time
    # the v2 loader must NOT auto-upgrade a bare anchor (no provenance / not a v2 envelope).
    with pytest.raises(CheckpointStampError):
        load_checkpoint(bare)


# ═══ Unregistered legacy encoding — loud raise, never shape-sniff ═════════════════════════════
def test_unregistered_legacy_encoding_raises(tmp_path, full_ls_state):
    """T-CK-34 — PASS iff a legacy read whose encoding_name is UNREGISTERED (the REAL full-v1 stamp
    'v6_live2', verified: lookup('v6_live2') → EncodingRegistryError, distinct from the registered
    'v6_live2_ls') raises LOUDLY at the registry lookup and NEVER falls back to inferring an arch
    from tensor shapes (the DELETED infer_*_hparams path). Bites: a silent shape-sniff fallback for
    an unregistered legacy encoding (resurrecting infer_model_hparams — the single most important
    untested KILL-resurrection surface)."""
    bare = tmp_path / "unregistered_legacy.pt"
    _save_bare(dict(full_ls_state), bare)  # a bare state dict that WOULD shape-sniff cleanly
    # 'v6_live2' is not in the registry → the ONLY correct behavior is a loud raise, never a sniff.
    with pytest.raises((EncodingRegistryError, RepresentationMismatch)):
        load_legacy_weights(bare, declared_encoding="v6_live2")
