"""test_anchor — the anchor consumer-of-record (WP10 §a.5/§c.6; LAW-08 CI consumer).

Exercises `save_best_model_atomic` round-trip verify + `.bak` rotation, `_quarantine_corrupt`,
the `load_best_model_resilient` fallback chain, and representation-off-the-declared-arch (a census
that `model_representation` is NOT imported/used — WP9 O3). PLUS the (B) corruption guard: a
checkpoint missing a required CORE key RAISES (never a silent random-head load — the old E1-C1 /
F-12 hazard), while a legitimate SUBSET anchor loads clean.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

import mantis.train.anchor as anchor_mod
from mantis.encoding import lookup
from mantis.model import arch_from_spec_and_config, build_net
from mantis.train.anchor import (
    AnchorLoadError,
    AnchorState,
    _quarantine_corrupt,
    load_best_model_resilient,
    save_best_model_atomic,
    state_dict_sha256,
)

_ANCHOR_SRC = Path(anchor_mod.__file__).read_text(encoding="utf-8")
_ENC = "v6_live2_ls"
_CPU = torch.device("cpu")
_OPTIONAL_PREFIXES = anchor_mod._OPTIONAL_HEAD_PREFIXES


def _full_net() -> torch.nn.Module:
    return build_net(arch_from_spec_and_config(lookup(_ENC), {}))


# ══ atomic save + .bak rotation + provenance ═══════════════════════════════════════════
def test_save_best_model_atomic_roundtrip_and_bak_rotation(tmp_path: Path) -> None:
    net = _full_net()
    path = tmp_path / "best_model.pt"

    save_best_model_atomic(net, path)  # bare state_dict (step=None)
    assert path.exists()
    assert not path.with_suffix(path.suffix + ".bak").exists()  # first save → no bak yet
    reloaded = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(reloaded, dict) and "trunk.input_conv.weight" in reloaded

    save_best_model_atomic(net, path)  # second save rotates the prior → .bak
    assert path.with_suffix(path.suffix + ".bak").exists()


def test_save_best_model_atomic_provenance_sidecar(tmp_path: Path) -> None:
    net = _full_net()
    path = tmp_path / "best_model.pt"
    save_best_model_atomic(net, path, step=750, run_id="runX", encoding=_ENC)
    wrapped = torch.load(path, map_location="cpu", weights_only=True)
    assert wrapped["step"] == 750 and wrapped["promoted"] is True
    assert wrapped["metadata"]["encoding_name"] == _ENC
    sidecar = path.with_name(path.name + ".provenance.json")
    assert sidecar.exists() and '"step": 750' in sidecar.read_text()


def test_quarantine_corrupt_renames(tmp_path: Path) -> None:
    path = tmp_path / "best_model.pt"
    path.write_bytes(b"not a torch file")
    dest = _quarantine_corrupt(path)
    assert not path.exists()
    assert dest.exists() and ".corrupt-" in dest.name


def test_state_dict_sha256_is_wrapper_invariant() -> None:
    net = _full_net()
    sd = net.state_dict()
    wrapped = {f"_orig_mod.{k}": v for k, v in sd.items()}
    assert state_dict_sha256(sd) == state_dict_sha256(wrapped)


# ══ resilient load fallback chain ══════════════════════════════════════════════════════
def test_load_best_model_resilient_loads_valid_anchor(tmp_path: Path) -> None:
    net = _full_net()
    path = tmp_path / "best_model.pt"
    save_best_model_atomic(net, path)

    ref = load_best_model_resilient(
        path, declared_encoding=_ENC, device=_CPU, bootstrap_candidates=(),
    )
    assert ref is not None
    model, source_path, _step, representation = ref
    assert isinstance(model, torch.nn.Module)
    assert source_path == path
    assert representation == "grid"  # declared, off the arch — not a module sniff


def test_load_best_model_resilient_recovers_from_bak(tmp_path: Path) -> None:
    net = _full_net()
    path = tmp_path / "best_model.pt"
    save_best_model_atomic(net, path)   # writes best
    save_best_model_atomic(net, path)   # rotates the valid copy into .bak
    path.write_bytes(b"corrupt zip")    # clobber the live best_model.pt

    ref = load_best_model_resilient(
        path, declared_encoding=_ENC, device=_CPU, bootstrap_candidates=(),
    )
    assert ref is not None
    _model, source_path, _step, _rep = ref
    assert source_path == path.with_suffix(path.suffix + ".bak")  # recovered from .bak
    assert any(".corrupt-" in p.name for p in tmp_path.iterdir())  # best was quarantined


def test_load_best_model_resilient_returns_none_when_all_fail(tmp_path: Path) -> None:
    path = tmp_path / "best_model.pt"  # absent; no .bak; empty bootstrap list
    assert load_best_model_resilient(
        path, declared_encoding=_ENC, device=_CPU, bootstrap_candidates=(),
    ) is None


# ══ representation off the DECLARED arch — no model_representation sniff (WP9 O3) ═══════
def test_no_model_representation_sniff_in_anchor_source() -> None:
    """Census (WP9 O3): the DELETED `model_representation(module)` sniff is NOT imported or used.
    Representation travels on the declared arch (`AnchorState.representation`)."""
    assert "model_representation" not in _ANCHOR_SRC


def test_no_pickle_exec_load_in_anchor_source() -> None:
    """LAW-12: anchor loads are weights-only everywhere (no pickle-exec load surface)."""
    assert "weights_only=False" not in _ANCHOR_SRC


def test_anchor_state_carries_declared_representation(tmp_path: Path) -> None:
    net = _full_net()
    path = tmp_path / "best_model.pt"
    save_best_model_atomic(net, path)
    ref = load_best_model_resilient(
        path, declared_encoding=_ENC, device=_CPU, bootstrap_candidates=(),
    )
    assert ref is not None
    model, source_path, step, representation = ref
    state = AnchorState(model, step, source_path, representation)
    assert state.representation == "grid"  # discriminant read off the arch


# ══ (B) corruption guard — the RED-TEAM #1 preserved landing guard ═════════════════════
def _core_and_optional_keys() -> "tuple[list[str], list[str]]":
    sd = _full_net().state_dict()
    optional = [k for k in sd if k.startswith(_OPTIONAL_PREFIXES)]
    core = [k for k in sd if not k.startswith(_OPTIONAL_PREFIXES)]
    return core, optional


def test_B_subset_anchor_missing_only_aux_heads_loads_clean(tmp_path: Path) -> None:
    """A legitimate SUBSET/min-max baseline anchor — the aux training-only heads absent, every
    CORE (trunk/policy/value) tensor present — loads clean (build_net emits a SUPERSET, so
    strict=True would spuriously reject; the (B) guard admits an optional-only subset)."""
    core, optional = _core_and_optional_keys()
    assert optional, "the full net must carry aux heads for this test to mean anything"
    full = _full_net().state_dict()
    subset = {k: v for k, v in full.items() if not k.startswith(_OPTIONAL_PREFIXES)}
    path = tmp_path / "subset_anchor.pt"
    torch.save(subset, path)  # bare state_dict (the T-CK-32 shape)

    ref = load_best_model_resilient(
        path, declared_encoding=_ENC, device=_CPU, bootstrap_candidates=(),
    )
    assert ref is not None  # clean load — no false reject on a legitimate subset


def test_B_missing_core_key_raises_not_silent_random_head(tmp_path: Path) -> None:
    """A checkpoint missing a REQUIRED CORE tensor (policy_fc.weight) RAISES — never a silent
    random-head load (the old E1-C1 / F-12 hazard the eval-loader landing-guard existed to kill)."""
    full = _full_net().state_dict()
    core_key = "policy_fc.weight"
    assert core_key in full and not core_key.startswith(_OPTIONAL_PREFIXES)
    corrupt = {k: v for k, v in full.items() if k != core_key}
    path = tmp_path / "core_missing_anchor.pt"
    torch.save(corrupt, path)

    with pytest.raises(AnchorLoadError, match="core"):
        load_best_model_resilient(
            path, declared_encoding=_ENC, device=_CPU, bootstrap_candidates=(),
        )
