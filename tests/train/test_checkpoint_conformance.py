# >300 justify (R8): an oracle-first conformance suite whose every test is numbered against a
# PREREG row (`T-CK-*`) and was written before the port existed. Splitting it breaks the 1:1
# test->spec mapping the design review reads it through, and the stamp/provenance/loader contract
# has one subject, not four.
"""Suite A — checkpoint CONFORMANCE (T-CK-01 … T-CK-33).

Written oracle-first against the envelope-v2 contract and the pinned old-side captures before any
port code, so each test's docstring cites its `T-CK-*` id and its one-line PASS bar and asserts on
PUBLIC surfaces only.

The resume-precedence tests T-CK-14/15/17 and the Trainer tests T-CK-18/19 import their symbols
LAZILY inside the test, so the other tests can go green before those modules land.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pytest
import torch

from mantis.config.loader import load_config
from mantis.config.schema import ARCH_SCOPED_KEYS
from mantis.encoding import EncodingRegistryError, all_specs
from mantis.model import GnnArch, RepresentationMismatch  # noqa: F401 (arch types)

import mantis.train.checkpoints as checkpoints


# WP11-A schema extension: eval.gate/eval.ladder are now required fields (design §c.1).
def _make_eval_block() -> dict:
    return {
        "random_model_sims": 96, "sealbot_model_sims": 128, "random_floor_games": 0, "worker_device": "cuda",
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

# The 18-key checkpoint-owned set, pinned LOCALLY, so T-CK-15's mutation self-test bites when the
# real constant is mutated.
_FROZEN_OWNED_LOCAL = frozenset({
    "encoding", "cluster_window_size", "cluster_threshold", "legal_move_radius", "board_size",
    "in_channels", "input_channels", "res_blocks", "filters", "se_reduction_ratio", "model",
    "total_steps", "scheduler_t_max", "eta_min", "min_lr", "lr", "weight_decay", "lr_schedule",
})


class _Evil:
    """A non-tensor picklable object — `weights_only=True` must refuse to unpickle it."""


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
    """Re-save a mutated payload under its correct `{run_id}_{step:08d}_{sha8}` name, so a stale content-hash never masks the field under test."""
    md = payload["metadata"]
    sha8 = content_sha8(payload)
    p = Path(checkpoint_dir) / checkpoint_filename(md["run_id"], md["step"], sha8)
    torch.save(payload, p)
    return p


def _overrides(result: Any) -> dict[str, Any]:
    """Accept either return shape of build_resume_config_overrides: a bare overrides dict, or a richer object with an `.overrides` field."""
    return dict(getattr(result, "overrides", result))


def test_full_envelope_has_v2_schema_fields(tmp_path, tiny_net, optim_scaler_sched,
                                            valid_config, metadata_kwargs):
    """T-CK-01 — the full payload carries schema_version==2, kind=='full', model_state, the three optimizer/scaler/scheduler states, config and the metadata block."""
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
    assert md.encoding_name == "gnn_axis_v1"
    assert md.run_id
    assert md.step == 100
    assert md.commit_sha
    assert md.created_utc
    assert isinstance(md.arch, GnnArch)
    assert hasattr(md, "corpus_sha256")


def test_weights_envelope_has_v2_schema_fields(tmp_path, tiny_net, optim_scaler_sched,
                                               valid_config, metadata_kwargs):
    """T-CK-02 — a weights save is kind=='weights' with model_state + metadata and NO optimizer/scaler/scheduler state."""
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
    """T-CK-03 — an invalid or incomplete config raises on write while a complete valid one saves."""
    opt, scaler, sched = optim_scaler_sched
    with pytest.raises(ValueError):  # pydantic ValidationError ⊂ ValueError
        _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                   config=invalid_config, meta=metadata_kwargs)
    good = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                      config=valid_config, meta=metadata_kwargs)
    assert good.exists()


def test_config_snapshot_schema_validated_on_read(tmp_path, tiny_net, optim_scaler_sched,
                                                  valid_config, metadata_kwargs):
    """T-CK-04 — loading an envelope whose embedded config fails schema raises, so the loader cannot skip config re-validation."""
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
    """T-CK-05 — a save whose encoding_name cannot be resolved raises; there is no metadata-omitted fallback."""
    opt, scaler, sched = optim_scaler_sched
    meta_no_enc = {"run_id": "runa", "arch": tiny_arch}  # encoding_name MISSING
    with pytest.raises(CheckpointStampError):
        _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                   config=valid_config, meta=meta_no_enc)


def test_filename_carries_run_id_step_sha8(tmp_path, tiny_net, optim_scaler_sched, valid_config,
                                           mk_meta, tiny_arch):
    """T-CK-06 — the basename is `{run_id}_{step:08d}_{sha8}.ckpt` and sha8 is the first 8 hex of the payload content hash."""
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
    """T-CK-07 — two distinct run_id at the SAME step produce distinct filenames, so the name cannot omit run_id."""
    opt, scaler, sched = optim_scaler_sched
    p1 = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                    config=valid_config, meta=mk_meta(tiny_arch, run_id="runa"), step=100)
    p2 = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                    config=valid_config, meta=mk_meta(tiny_arch, run_id="runb"), step=100)
    assert p1.name != p2.name
    assert p1.name.startswith("runa_00000100_")
    assert p2.name.startswith("runb_00000100_")


def test_load_reverifies_run_id_and_step(tmp_path, tiny_net, optim_scaler_sched, valid_config,
                                         mk_meta, tiny_arch):
    """T-CK-08 — loading a file whose embedded run_id/step disagree with the filename raises: the loader never trusts the filename."""
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
    """T-CK-09 — mutating one model_state element with sha8 unchanged in the name makes load raise a content-hash mismatch."""
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


def test_restamp_from_loaded_config_is_error(tmp_path, tiny_net, optim_scaler_sched, valid_config,
                                             metadata_kwargs):
    """T-CK-10 — re-saving a loaded envelope while re-deriving metadata from the loaded config raises: stamps are minted ONCE by save, so supplying created_utc/commit_sha is refused."""
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


def test_unstamped_save_fails_loud_and_writes_nothing(tmp_path, tiny_net, optim_scaler_sched,
                                                      valid_config, tiny_arch):
    """T-CK-11 — an unstampable save raises AND leaves no canonical file."""
    opt, scaler, sched = optim_scaler_sched
    unstampable = {"run_id": "runa", "arch": tiny_arch}  # no encoding_name → cannot stamp
    with pytest.raises(CheckpointStampError):
        _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                   config=valid_config, meta=unstampable, allow_quarantine=False)
    assert list(tmp_path.glob("*.ckpt")) == []


def test_quarantine_path_when_run_must_survive(tmp_path, tiny_net, optim_scaler_sched, valid_config,
                                               tiny_arch, monkeypatch):
    """T-CK-12 — with the survive-run flag an unstampable save writes `<path>.quarantine` and increments the QUARANTINE counter, not the persist-fatal one, and never a canonical name."""
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


def test_every_load_surface_uses_weights_only_true(tmp_path, tiny_net, optim_scaler_sched,
                                                   valid_config, metadata_kwargs):
    """T-CK-13 — a checkpoint carrying a non-tensor picklable object fails to load, and a source census finds ZERO `weights_only=False` surfaces."""
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


def test_launch_config_wins_except_frozen_keys():
    """T-CK-14 — a non-frozen override is applied while a frozen key (encoding/arch/optim/sched) defers to the checkpoint."""
    from mantis.train.orchestrator import build_resume_config_overrides
    baked = {"lr": 0.001, "encoding": "gnn_axis_v1", "log_interval": 500}
    launch = {"lr": 0.002, "encoding": "gnn_axis_r8", "log_interval": 250}
    ov = _overrides(build_resume_config_overrides(baked, launch))
    assert ov.get("log_interval") == 250
    assert "lr" not in ov
    assert "encoding" not in ov


def test_frozen_key_set_is_the_pinned_constant(resume_goldens):
    """T-CK-15 — RESUME_CHECKPOINT_OWNED_KEYS equals the exact 18-key set, and mutating the constant bites."""
    from mantis.train.orchestrator import RESUME_CHECKPOINT_OWNED_KEYS
    golden = resume_goldens["T-CK-15_frozen_key_set"]
    assert set(RESUME_CHECKPOINT_OWNED_KEYS) == set(golden["sorted_keys"])
    assert len(RESUME_CHECKPOINT_OWNED_KEYS) == golden["count"] == 18
    assert frozenset(RESUME_CHECKPOINT_OWNED_KEYS) == _FROZEN_OWNED_LOCAL


def test_declared_key_wins_base_inherited_defers(resume_goldens, spy_sink):
    """T-CK-16 — a declared key wins over baked, a base-inherited key defers to baked and warns on a difference, and a declared null travels."""
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


def test_scheduler_horizon_gate(resume_goldens):
    """T-CK-17 — without the flag the horizon keys stay OWNED; with it, total_steps and scheduler_t_max re-enter the overrides."""
    from mantis.train.orchestrator import build_resume_config_overrides
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
    """T-CK-18 — a full resume with scheduler_state None raises unless allow_fresh_scheduler, which warns instead."""
    from mantis.train.trainer.core import Trainer
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
    """T-CK-19 — a full resume restores optimizer (param_groups==2), scaler and step==750, while a weights resume gets a fresh optimizer and step==500."""
    from mantis.train.trainer.core import Trainer
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
    """T-CK-20 — a declared lr differing from the baked lr on a full resume is IGNORED, since lr is resume-state-owned, and the ignore is loud."""
    g = resume_goldens["T-CK-20_lr_resume_owned"]["expected_output"]
    ign = g["resolve_lr_provenance_override_ignored_case"]
    prov = resolve_lr_provenance(declared=ign["declared"], baked=ign["baked"],
                                 effective=ign["effective"])
    assert prov.override_ignored is True
    norm = resolve_lr_provenance(declared=0.001, baked=0.001, effective=0.001)
    assert norm.override_ignored is g["resolve_lr_provenance_normal_case_declared_eq_baked"]["override_ignored"]


def test_weights_strip_requires_wire_signature_equality(tmp_path, tiny_net, optim_scaler_sched,
                                                        valid_config, metadata_kwargs):
    """T-CK-21 — the weights-strip and re-stamp succeeds only on wire-signature equality, and a mismatch raises."""
    opt, scaler, sched = optim_scaler_sched
    src = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                     config=valid_config, meta=metadata_kwargs)
    same = strip_and_restamp(src, new_encoding="gnn_axis_v1", run_id="runc",
                             checkpoint_dir=tmp_path)
    assert Path(same).exists()
    # THE REFUSAL ARM HAS NO CONSTRUCTIBLE INPUT AT HEAD, asserted rather than quietly dropped:
    # every registered encoding now shares one wire signature, so no `new_encoding` this repo knows
    # can make the check fire. The row below reds the day a second signature is registered.
    signatures = {checkpoints._wire_signature(spec) for spec in all_specs()}
    assert len(signatures) == 1, (
        f"more than one registered wire signature ({signatures}) — the strip's mismatch "
        "refusal is now constructible and needs its arm back")


def _save_bare(state: dict, path: Path) -> None:
    torch.save(state, path)


def _inject(state: dict, prefix: str) -> dict:
    dirty = dict(state)
    dirty[f"{prefix}some.weight"] = torch.zeros(1)
    return dirty


def _forge_v2_with_killed_prefix(tmp_path, *, net, opt, scaler, sched, config, meta,
                                 prefix: str) -> Path:
    """Build a REAL stamped v2 envelope, hand-forge a killed prefix into its model_state, and re-name it so the provenance and content-hash checks pass and the killed-prefix reject is what fires."""
    good = _save_full(tmp_path, net=net, opt=opt, scaler=scaler, sched=sched, config=config,
                      meta=meta)
    payload = _load_raw(good)
    payload["model_state"][f"{prefix}some.weight"] = torch.zeros(1)
    return _resave_rehashed(payload, tmp_path)


def test_reject_cluster_pool_prefix(tmp_path, full_graph_state, tiny_net, optim_scaler_sched,
                                    valid_config, metadata_kwargs):
    """T-CK-22 — a synthetic `cluster_pool.` key makes BOTH loader surfaces raise RepresentationMismatch, so no PMA pool is sniff-reconstructed."""
    opt, scaler, sched = optim_scaler_sched
    p = tmp_path / "dirty_cluster.pt"
    _save_bare(_inject(full_graph_state, "cluster_pool."), p)
    with pytest.raises(RepresentationMismatch):
        load_legacy_weights(p, declared_encoding="gnn_axis_v1")
    forged = _forge_v2_with_killed_prefix(tmp_path, net=tiny_net, opt=opt, scaler=scaler,
                                          sched=sched, config=valid_config, meta=metadata_kwargs,
                                          prefix="cluster_pool.")
    with pytest.raises(RepresentationMismatch):
        load_checkpoint(forged)


def test_reject_global_encoder_prefix(tmp_path, full_graph_state, tiny_net, optim_scaler_sched,
                                      valid_config, metadata_kwargs):
    """T-CK-23 — a `global_encoder.` key raises on BOTH surfaces, so pma_global is never reconstructed."""
    opt, scaler, sched = optim_scaler_sched
    p = tmp_path / "dirty_global.pt"
    _save_bare(_inject(full_graph_state, "global_encoder."), p)
    with pytest.raises(RepresentationMismatch):
        load_legacy_weights(p, declared_encoding="gnn_axis_v1")
    forged = _forge_v2_with_killed_prefix(tmp_path, net=tiny_net, opt=opt, scaler=scaler,
                                          sched=sched, config=valid_config, meta=metadata_kwargs,
                                          prefix="global_encoder.")
    with pytest.raises(RepresentationMismatch):
        load_checkpoint(forged)


def test_reject_gpool_bias_branch_prefix(tmp_path, full_graph_state, tiny_net, optim_scaler_sched,
                                         valid_config, metadata_kwargs):
    """T-CK-24 — a `gpool_bias_branch.` key raises on BOTH surfaces, so the gpool bias is never reconstructed."""
    opt, scaler, sched = optim_scaler_sched
    p = tmp_path / "dirty_gpool.pt"
    _save_bare(_inject(full_graph_state, "gpool_bias_branch."), p)
    with pytest.raises(RepresentationMismatch):
        load_legacy_weights(p, declared_encoding="gnn_axis_v1")
    forged = _forge_v2_with_killed_prefix(tmp_path, net=tiny_net, opt=opt, scaler=scaler,
                                          sched=sched, config=valid_config, meta=metadata_kwargs,
                                          prefix="gpool_bias_branch.")
    with pytest.raises(RepresentationMismatch):
        load_checkpoint(forged)


def test_clean_anchor_loads(tmp_path, full_graph_net, full_graph_state, anchor_key_set):
    """T-CK-25 — the committed anchor key set is a subset of the stripped `build_net` key set and loads clean, so a clean promoted anchor is never falsely rejected."""
    constructed = set(full_graph_net.state_dict().keys())
    assert anchor_key_set
    assert anchor_key_set <= constructed
    assert not any(k.startswith(KILLED_PREFIXES) for k in anchor_key_set)
    p = tmp_path / "clean_anchor.pt"
    _save_bare(full_graph_state, p)
    ck = load_legacy_weights(p, declared_encoding="gnn_axis_v1")
    assert ck.model_state


def test_killed_prefix_reject_mutation_selftest(tmp_path, full_graph_state):
    """T-CK-26 — a clean state dict loads AND injecting a killed prefix makes the loader reject, so the guard is wired and fires."""
    clean = tmp_path / "clean.pt"
    _save_bare(dict(full_graph_state), clean)
    assert load_legacy_weights(clean, declared_encoding="gnn_axis_v1").model_state
    dirty = tmp_path / "dirty.pt"
    _save_bare(_inject(full_graph_state, "cluster_pool."), dirty)
    with pytest.raises(RepresentationMismatch):
        load_legacy_weights(dirty, declared_encoding="gnn_axis_v1")


def test_declared_encoding_mismatch_raises(tmp_path, tiny_net, optim_scaler_sched, valid_config,
                                           metadata_kwargs):
    """T-CK-27 — a declared_encoding disagreeing with the stamp raises DeclaredEncodingMismatchError naming both."""
    opt, scaler, sched = optim_scaler_sched
    path = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                      config=valid_config, meta=metadata_kwargs)
    with pytest.raises(DeclaredEncodingMismatchError):
        load_checkpoint(path, declared_encoding="gnn_axis_r8")


def test_decode_override_wins_and_logs_never_raises(tmp_path, tiny_net, optim_scaler_sched,
                                                    valid_config, metadata_kwargs, caplog):
    """T-CK-28 — decode_override is authoritative and logs loudly on disagreement, but NEVER raises."""
    import logging
    opt, scaler, sched = optim_scaler_sched
    path = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                      config=valid_config, meta=metadata_kwargs)
    with caplog.at_level(logging.INFO):  # floor at INFO so a WARNING-or-INFO notice is captured
        ck = load_checkpoint(path, decode_override="gnn_axis_r8")
    assert ck is not None
    assert "encoding_decode_override" in caplog.text


def test_declared_and_override_together_error(tmp_path, tiny_net, optim_scaler_sched, valid_config,
                                              metadata_kwargs):
    """T-CK-29 — passing both declared_encoding and decode_override raises ValueError: the two are mutually exclusive."""
    opt, scaler, sched = optim_scaler_sched
    path = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                      config=valid_config, meta=metadata_kwargs)
    with pytest.raises(ValueError):
        load_checkpoint(path, declared_encoding="gnn_axis_v1", decode_override="gnn_axis_r8")


def test_stamp_sources_disagree_raises(tmp_path, tiny_net, optim_scaler_sched, valid_config,
                                       metadata_kwargs):
    """T-CK-30 — a checkpoint whose metadata.encoding_name and config.encoding resolve to DIFFERENT names raises rather than silently picking one source."""
    opt, scaler, sched = optim_scaler_sched
    path = _save_full(tmp_path, net=tiny_net, opt=opt, scaler=scaler, sched=sched,
                      config=valid_config, meta=metadata_kwargs)
    payload = _load_raw(path)
    payload["config"]["identity"]["encoding"] = "gnn_axis_r8"  # config and metadata now disagree
    bad = _resave_rehashed(payload, tmp_path)
    with pytest.raises(CheckpointStampError):
        load_checkpoint(bad)


def test_reads_full_v1_envelope_via_field_map(tmp_path, full_graph_net, full_graph_state, legacy_shapes):
    """T-CK-31 — a full old envelope reads via the old→v2 field map (training_date→created_utc, model_architecture/variant→arch, train_config_path dropped, config re-validated) to a resume-capable load."""
    fv1 = legacy_shapes["full_v1_envelope"]
    md = fv1["metadata"]
    from mantis.model import arch_from_spec_and_config
    from mantis.encoding import all_specs, lookup
    opt = torch.optim.AdamW(full_graph_net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000000, eta_min=0.0005)
    valid_config = {
        "schema_version": 1, "run_id": "run5", "seed": 20260718,
        "eval_enabled": True,
        # A REQUIRED top-level leaf whose `null` is the placeholder: refused at boot on a cuda
        # process, and valued only by the re-calibration sitting.
        "allocator_posture": None,
        "identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
        "eval": _make_eval_block(),
        # DERIVED from a MINTED config, not a restatement of the complete `train:` block.
        "train": load_config(
            Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml"
        ).train.model_dump(),
        "search": {"kind": "puct"},
        "selfplay": {
            "n_workers": 1, "leaf_batch_size": 8, "max_game_moves": 128,
            "c_visit": 50.0,
            "c_scale": 1.0, "gumbel_m": 16, "gumbel_explore_moves": 10,
            "results_queue_cap": 10_000, "random_opening_plies": 0,
            "log_investigation_metrics": True,
            "mcts": {"n_simulations": 50, "c_puct": 1.5, "fpu_reduction": 0.25,
                     "quiescence_enabled": True, "quiescence_blend_2": 0.3,
                     "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25,
                     "dirichlet_enabled": True},
            "playout_cap": {"fast_sims": 50, "fast_prob": 0.0, "standard_sims": 0,
                            "full_search_prob": 0.0, "n_sims_quick": 0, "n_sims_full": 0,
                            "temperature_threshold_compound_moves": 0, "temp_min": 0.5},
        },
        "inference": {
            "inference_batch_size": 64, "inference_max_wait_ms": 10,
            # ARCH-SCOPED to graph and this envelope IS a graph one, so it is REQUIRED here; the
            # pair is the template's non-binding value.
            "fused_graph_caps": {"max_fused_edges": 57149441, "max_fused_nodes": 1785921},
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
    # Every ARCH-SCOPED block that is not this envelope's arch is dropped through
    # `ARCH_SCOPED_KEYS` — the schema's own partition — rather than by name, so a third scoped
    # block needs no edit here.
    for _key in ARCH_SCOPED_KEYS:
        if valid_config["identity"]["representation"] != _key.arch:
            valid_config[_key.section].pop(_key.field, None)
    payload = {  # the real FULL-v1 top-level shape (7 keys) + captured metadata scalars
        "step": fv1["step"],
        "model_state": full_graph_state,
        "optimizer_state": opt.state_dict(),
        "scaler_state": torch.amp.GradScaler("cpu", enabled=True).state_dict(),
        "scheduler_state": sched.state_dict(),
        "config": valid_config,
        "metadata": {
            "encoding_name": "gnn_axis_v1",             # registered (arch-resolvable); verbatim-passthrough
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
    assert ck.kind == "full"
    assert ck.optimizer_state is not None
    assert ck.scheduler_state is not None
    assert ck.metadata.encoding_name == "gnn_axis_v1"
    assert ck.metadata.commit_sha == md["commit_sha"]
    assert ck.metadata.created_utc == md["training_date"]  # training_date → created_utc, VERBATIM
    assert ck.metadata.corpus_sha256 is None
    assert not hasattr(ck.metadata, "train_config_path")   # dropped (not a v2 metadata field)
    assert not ck.metadata.run_id                        # SYNTHESIZED-NEVER on a legacy read
    assert isinstance(ck.metadata.arch, GnnArch)


def test_reads_bare_state_dict_anchor_no_fake_provenance(tmp_path, full_graph_state, legacy_shapes):
    """T-CK-32 — a BARE state_dict anchor loads via load_legacy_weights with the arch taken from the declared encoding, kind='weights', and NO synthetic run_id/hash/created_utc."""
    assert legacy_shapes["bare_state_dict"]["top_level_is_envelope"] is False
    bare = tmp_path / "bootstrap_model_v6_live2.pt"
    torch.save(full_graph_state, bare)  # the whole payload IS the state dict — no wrapper
    ck = load_legacy_weights(bare, declared_encoding="gnn_axis_v1")
    assert ck.kind == "weights"
    assert ck.model_state
    assert ck.metadata.encoding_name == "gnn_axis_v1"
    assert not ck.metadata.run_id
    assert not getattr(ck.metadata, "created_utc", "")
    assert ck.optimizer_state is None


def test_bare_anchor_to_v2_requires_explicit_strip(tmp_path, full_graph_state):
    """T-CK-33 — upgrading a legacy bare anchor to a stamped v2 envelope goes ONLY through the wire-signature-gated weights-only strip and re-stamp; an auto-restamp on read raises."""
    bare = tmp_path / "bootstrap_model_v6_live2.pt"
    torch.save(full_graph_state, bare)
    # sanctioned upgrade: strip + re-stamp → a proper v2 envelope with a FRESH single stamp.
    v2_path = strip_and_restamp(bare, new_encoding="gnn_axis_v1", run_id="runx",
                                checkpoint_dir=tmp_path, declared_encoding="gnn_axis_v1")
    ck = load_checkpoint(v2_path)
    assert ck.metadata.run_id == "runx"
    assert ck.metadata.created_utc
    # the v2 loader must NOT auto-upgrade a bare anchor (no provenance / not a v2 envelope).
    with pytest.raises(CheckpointStampError):
        load_checkpoint(bare)


def test_unregistered_legacy_encoding_raises(tmp_path, full_graph_state):
    """T-CK-34 — a legacy read whose encoding_name is UNREGISTERED raises LOUDLY at the registry lookup and NEVER falls back to inferring an arch from tensor shapes."""
    bare = tmp_path / "unregistered_legacy.pt"
    _save_bare(dict(full_graph_state), bare)  # a bare state dict that WOULD shape-sniff cleanly
    # 'v6_live2' is not in the registry → the ONLY correct behavior is a loud raise, never a sniff.
    with pytest.raises((EncodingRegistryError, RepresentationMismatch)):
        load_legacy_weights(bare, declared_encoding="v6_live2")
