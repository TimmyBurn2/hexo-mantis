"""The anchor's consumer-of-record: save, load, quarantine and identity.

>300 justify (R8): these are ONE artifact lifecycle and mean nothing apart — split, a change can
pass both files while breaking the pair, the write dropping the arch and the read inventing one.
"""
from __future__ import annotations

import dataclasses
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
)

_ANCHOR_SRC = Path(anchor_mod.__file__).read_text(encoding="utf-8")
_ENC = "gnn_axis_v1"
_CPU = torch.device("cpu")
_OPTIONAL_PREFIXES = anchor_mod._OPTIONAL_HEAD_PREFIXES


def _full_net() -> torch.nn.Module:
    return build_net(arch_from_spec_and_config(lookup(_ENC), {}))


def test_save_best_model_atomic_roundtrip_and_bak_rotation(tmp_path: Path) -> None:
    net = _full_net()
    path = tmp_path / "best_model.pt"

    save_best_model_atomic(net, path)  # bare state_dict
    assert path.exists()
    assert not path.with_suffix(path.suffix + ".bak").exists()  # first save, no bak yet
    reloaded = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(reloaded, dict) and "representation.input_proj.weight" in reloaded

    save_best_model_atomic(net, path)  # the second save rotates the prior into .bak
    assert path.with_suffix(path.suffix + ".bak").exists()


def test_save_best_model_atomic_provenance_sidecar(tmp_path: Path) -> None:
    net = _full_net()
    arch = arch_from_spec_and_config(lookup(_ENC), {})
    path = tmp_path / "best_model.pt"
    net.arch = arch
    save_best_model_atomic(net, path, step=750, run_id="runX", encoding=_ENC)
    wrapped = torch.load(path, map_location="cpu", weights_only=True)
    assert wrapped["step"] == 750 and wrapped["promoted"] is True
    assert wrapped["metadata"]["encoding_name"] == _ENC
    sidecar = path.with_name(path.name + ".provenance.json")
    assert sidecar.exists() and '"step": 750' in sidecar.read_text()


def test_a_promoted_anchor_carries_the_DECLARED_arch_kind(tmp_path: Path) -> None:
    """The payload says WHICH arch built it, or the read side rebuilds the INCUMBENT kind."""
    arch = arch_from_spec_and_config(lookup(_ENC), {})
    path = tmp_path / "best_model.pt"
    net = _full_net()
    net.arch = arch
    save_best_model_atomic(net, path, step=10, run_id="r", encoding=_ENC)
    stamped = torch.load(path, map_location="cpu", weights_only=True)["metadata"]["arch"]
    assert stamped["arch_kind"] == type(arch).__name__
    assert stamped["representation"] == arch.representation


def test_a_promotion_that_cannot_NAME_its_arch_is_REFUSED(tmp_path: Path) -> None:
    """PLANTED BREAK: a stamped save with no arch is unconstructible, not silently written."""
    handle_less = _full_net()
    del handle_less.arch
    with pytest.raises(AttributeError, match="declared '.arch'"):
        save_best_model_atomic(
            handle_less, tmp_path / "best_model.pt", step=10, run_id="r", encoding=_ENC,
        )


def test_a_NON_INCUMBENT_lineage_survives_a_save_load_round_trip(tmp_path: Path) -> None:
    """A non-incumbent lineage survives a round trip through the production loader, rebuilding
    as its own kind and not into a quarantine. With no kind in the payload the shape load
    quarantines, losing that lineage's promoted incumbent on EVERY relaunch."""
    from mantis.encoding import lookup as _lookup
    from mantis.model import ARCH_KINDS, INCUMBENT_ARCH_KIND, select_arch

    spec = _lookup("gnn_axis_v1")
    incumbent = INCUMBENT_ARCH_KIND[spec.representation]
    others = [k for k in ARCH_KINDS if k != incumbent and k.startswith("Gnn")]
    assert others, "no non-incumbent graph kind exists — this row would be vacuous"
    kind = others[0]
    arch = select_arch(spec, {}, arch_kind=kind)
    net = build_net(arch)

    path = tmp_path / "best_model.pt"
    save_best_model_atomic(net, path, step=42, run_id="r", encoding=spec.name)
    loaded = load_best_model_resilient(
        path, declared_encoding=spec.name, device=_CPU, bootstrap_candidates=(),
    )
    assert loaded is not None, "the anchor was not loadable at all"
    model, source, step, representation = loaded
    expected_net_type = type(build_net(select_arch(spec, {}, arch_kind=kind)))
    incumbent_net_type = type(build_net(select_arch(spec, {}, arch_kind=incumbent)))
    assert expected_net_type is not incumbent_net_type, (
        "the two kinds build the same net type, so this row could not tell them apart"
    )
    assert type(model) is expected_net_type, (
        f"the anchor rebuilt as {type(model).__name__}, not the {kind} net it was written from"
    )
    assert source == path, "fell through to a fallback — the live anchor did not load"
    assert step == 42 and representation == spec.representation
    assert not list(path.parent.glob("*.corrupt-*")), (
        "a valid non-incumbent anchor was QUARANTINED — F-17's exact failure"
    )


def test_an_anchor_with_NON_DEFAULT_widths_rebuilds_at_ITS_widths(tmp_path: Path) -> None:
    """An anchor at non-default widths rebuilds at ITS widths, which is why the stamped arch is
    rehydrated VERBATIM: re-deriving takes widths from a config an anchor does not carry.
    PLANTED BREAK: rehydrate through `select_arch(spec, {}, ...)` and the shape load reds."""
    from mantis.encoding import lookup as _lookup
    from mantis.model import select_arch

    spec = _lookup("gnn_axis_v1")
    default_arch = select_arch(spec, {}, arch_kind="GnnArch")
    narrow = dataclasses.replace(default_arch, hidden=default_arch.hidden // 2)
    assert narrow.hidden != default_arch.hidden, "the fixture must differ from the default"

    path = tmp_path / "best_model.pt"
    save_best_model_atomic(build_net(narrow), path, step=7, run_id="r", encoding=spec.name)
    loaded = load_best_model_resilient(
        path, declared_encoding=spec.name, device=_CPU, bootstrap_candidates=(),
    )
    assert loaded is not None, "an anchor at a non-default width did not load at all"
    model = loaded[0]
    assert model.arch.hidden == narrow.hidden, (
        f"rebuilt at hidden={model.arch.hidden} but the anchor was written at "
        f"{narrow.hidden} — the widths came from the dataclass defaults, not the artifact"
    )
    assert not list(path.parent.glob("*.corrupt-*"))


def test_a_WRONG_ARCH_anchor_RAISES_rather_than_being_quarantined(tmp_path: Path) -> None:
    """A wrong-arch anchor RAISES rather than quarantining: the artifact is intact and the arch
    is wrong, a configuration error an operator must SEE. Built by stamping a NARROW net under a
    DEFAULT-width arch, so the rebuild is wrong-shaped while the file reads perfectly."""
    from mantis.encoding import lookup as _lookup
    from mantis.model import select_arch

    spec = _lookup("gnn_axis_v1")
    default_arch = select_arch(spec, {}, arch_kind="GnnArch")
    narrow = dataclasses.replace(default_arch, hidden=default_arch.hidden // 2)

    path = tmp_path / "best_model.pt"
    save_best_model_atomic(build_net(narrow), path, step=3, run_id="r", encoding=spec.name)
    # Claim the DEFAULT widths in the stamp while the tensors stay narrow.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["metadata"]["arch"] = {
        **dataclasses.asdict(default_arch), "arch_kind": type(default_arch).__name__,
    }
    torch.save(payload, path)

    # `RuntimeError` covers both arms deliberately: `AnchorLoadError` subclasses it, and a
    # pure shape mismatch arrives as a bare one. The claim is that NEITHER is swallowed.
    with pytest.raises(RuntimeError, match="size mismatch|did not land|unexpected key"):
        load_best_model_resilient(
            path, declared_encoding=spec.name, device=_CPU, bootstrap_candidates=(),
        )
    assert not list(path.parent.glob("*.corrupt-*")), (
        "an intact anchor whose ARCH is wrong was quarantined — the bare-except defect"
    )


def test_a_GENUINELY_corrupt_anchor_is_still_quarantined(tmp_path: Path) -> None:
    """Control for the row above: a file that is not a torch archive still quarantines."""
    path = tmp_path / "best_model.pt"
    path.write_bytes(b"not a torch file at all")
    assert load_best_model_resilient(
        path, declared_encoding=_ENC, device=_CPU, bootstrap_candidates=(),
    ) is None
    assert list(path.parent.glob("*.corrupt-*")), "a truly corrupt anchor must be quarantined"


def test_quarantine_corrupt_renames(tmp_path: Path) -> None:
    path = tmp_path / "best_model.pt"
    path.write_bytes(b"not a torch file")
    dest = _quarantine_corrupt(path)
    assert not path.exists()
    assert dest.exists() and ".corrupt-" in dest.name


def test_the_anchor_hash_is_wrapper_invariant_and_is_THE_net_param_hash() -> None:
    """The anchor's identity is wrapper-invariant AND is the module-level `net_param_hash` —
    a separate anchor-only hash is uncomparable with what a sweep or a witness reports."""
    from mantis.model.identity import net_param_hash, state_dict_param_hash

    net = _full_net()
    sd = net.state_dict()
    wrapped = {f"_orig_mod.{k}": v for k, v in sd.items()}
    assert state_dict_param_hash(sd) == state_dict_param_hash(wrapped)
    assert state_dict_param_hash(sd) == net_param_hash(net), (
        "the state-dict entry point and the module entry point must be ONE hash — that they "
        "were not is exactly what made the launch pin uncomparable to every recorded observable"
    )


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
    assert representation == "graph"  # declared, off the arch, not a module sniff


def test_load_best_model_resilient_recovers_from_bak(tmp_path: Path) -> None:
    net = _full_net()
    path = tmp_path / "best_model.pt"
    save_best_model_atomic(net, path)   # writes best
    save_best_model_atomic(net, path)   # rotates the valid copy into .bak
    path.write_bytes(b"corrupt zip")    # clobbers the live best_model.pt

    ref = load_best_model_resilient(
        path, declared_encoding=_ENC, device=_CPU, bootstrap_candidates=(),
    )
    assert ref is not None
    _model, source_path, _step, _rep = ref
    assert source_path == path.with_suffix(path.suffix + ".bak")
    assert any(".corrupt-" in p.name for p in tmp_path.iterdir())  # best was quarantined


def test_load_best_model_resilient_returns_none_when_all_fail(tmp_path: Path) -> None:
    path = tmp_path / "best_model.pt"  # absent; no .bak; empty bootstrap list
    assert load_best_model_resilient(
        path, declared_encoding=_ENC, device=_CPU, bootstrap_candidates=(),
    ) is None


def test_no_model_representation_sniff_in_anchor_source() -> None:
    """The deleted `model_representation` sniff is neither imported nor used."""
    assert "model_representation" not in _ANCHOR_SRC


def test_no_pickle_exec_load_in_anchor_source() -> None:
    """Anchor loads are weights-only everywhere — no pickle-exec load surface."""
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
    assert state.representation == "graph"  # the discriminant, read off the arch


def _core_and_optional_keys() -> "tuple[list[str], list[str]]":
    sd = _full_net().state_dict()
    optional = [k for k in sd if k.startswith(_OPTIONAL_PREFIXES)]
    core = [k for k in sd if not k.startswith(_OPTIONAL_PREFIXES)]
    return core, optional


def test_B_missing_core_key_raises_not_silent_random_head(tmp_path: Path) -> None:
    """A checkpoint missing a REQUIRED CORE tensor RAISES, never a silent random-head load.
    The optional-subset twin is gone: those prefixes name dense aux heads a graph net lacks."""
    full = _full_net().state_dict()
    core_key = "policy_head.mlp.0.weight"
    assert core_key in full and not core_key.startswith(_OPTIONAL_PREFIXES)
    corrupt = {k: v for k, v in full.items() if k != core_key}
    path = tmp_path / "core_missing_anchor.pt"
    torch.save(corrupt, path)

    with pytest.raises(AnchorLoadError, match="core"):
        load_best_model_resilient(
            path, declared_encoding=_ENC, device=_CPU, bootstrap_candidates=(),
        )
