"""AUDIT-1 F-32 / R334(c) SHAPE A, ARMED at the run6 mint (R338) — the launch pin's TWO ends.

THE DEFECT THIS FILE EXISTS FOR, and it is the mirror of the one the audit found. The audit
recorded `verify_launch_anchor_pin` as *"a refusal nobody can reach"*: nothing passed
`expected_anchor_sha256`, so the guard was inert. Arming it from `identity.warm_start` and
changing nothing else makes it a refusal nobody can PASS — because the guard's other input,
`getattr(trainer, "checkpoint_source", None)`, was set by NOTHING in the tree, and the guard
FAILS CLOSED when a pin is set and no source is readable. Run6's first launch has no
`best_model.pt` and none of `_BOOTSTRAP_ANCHOR_CANDIDATES`, so it takes exactly that branch.
A guard that refuses every launch is not stricter than one that never fires; it is broken in
the other direction, and both directions ship green if only one end is tested.

SO BOTH ENDS ARE TESTED HERE, against the same row:
  * the PIN end — `mantis.run` derives `expected_anchor_sha256` from `identity.warm_start`,
    one source, no hand-synced twin (R334(c));
  * the SOURCE end — `init_trainer`'s fresh branch sets `checkpoint_source` from that same
    row, so the artifact the guard hashes is the one R336(d) calls the step-0 anchor.

AND THE EQUALITY BETWEEN THEM IS MEASURED, NOT ASSUMED. `warm_start.net_hash` is
`net_param_hash` over a net REBUILT from the artifact's stamp; the guard's side is
`checkpoint_state_sha256` — `state_dict_param_hash` over the STORED state. F-32 unified the
DENOMINATION and the docstrings say so, but "the same function" is not "the same value on this
artifact": a buffer the rebuild adds, or one the save drops, moves one side and not the other.
The row below writes a real checkpoint through the ONE writer and asserts the two agree on it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from mantis.encoding import lookup
from mantis.model import build_net, select_arch
from mantis.model.identity import net_param_hash
from mantis.train.anchor import checkpoint_state_sha256, verify_launch_anchor_pin

_ENC = "gnn_axis_v1"


def _arch() -> Any:
    """A deliberately NARROW graph arch — small enough to build fast, and off the dataclass
    defaults so a rebuild that ignored the stamp would be visible."""
    import dataclasses

    base = select_arch(lookup(_ENC), {}, arch_kind="GnnArch")
    return dataclasses.replace(base, hidden=32, num_layers=2, policy_hidden=32, value_hidden=16)


def _write_source(tmp_path: Path, arch: Any, *, run_id: str = "bcsrc") -> tuple[Path, str]:
    """A BC-shaped source checkpoint through the ONE writer, plus its `net_param_hash`."""
    from mantis.train.checkpoints import save_checkpoint

    from _warmstart_config import minimal_config  # noqa: PLC0415

    net = build_net(arch)
    path = save_checkpoint(
        model=net, optimizer=None, scaler=None, scheduler=None, step=0,
        config=minimal_config(), kind="weights",
        metadata_kwargs={"encoding_name": _ENC, "run_id": run_id, "arch": arch},
        checkpoint_dir=tmp_path,
    )
    return Path(path), net_param_hash(net)


def _config_with_row(checkpoint: Path, net_hash: str) -> dict[str, Any]:
    from _warmstart_config import minimal_config  # noqa: PLC0415

    cfg = minimal_config()
    cfg["identity"]["encoding"] = _ENC
    cfg["identity"]["representation"] = "graph"
    cfg["identity"]["warm_start"] = {"checkpoint": str(checkpoint), "net_hash": net_hash}
    return cfg


# ══ the guard's own contract, both directions ══════════════════════════════════════════
def test_a_pinned_fresh_init_REFUSES_when_no_source_is_readable() -> None:
    """The half that makes the wiring necessary. This is not a hypothetical: it is what an
    armed pin did on EVERY fresh launch before `checkpoint_source` had a producer."""
    with pytest.raises(RuntimeError, match="no readable --checkpoint"):
        verify_launch_anchor_pin(expected_anchor_sha256="a" * 64, checkpoint_path=None,
                                 trainer_step=0, run_id="run6")


def test_an_unpinned_fresh_init_is_a_NO_OP_with_or_without_a_source() -> None:
    """The pre-row posture is preserved exactly: no pin, no opinion."""
    verify_launch_anchor_pin(expected_anchor_sha256=None, checkpoint_path=None,
                             trainer_step=0, run_id="r")
    verify_launch_anchor_pin(expected_anchor_sha256=None, checkpoint_path="/nonexistent.pt",
                             trainer_step=0, run_id="r")


def test_a_pinned_fresh_init_PASSES_against_the_artifact_the_row_names(tmp_path: Path) -> None:
    """The wired shape, end to end: pin from the row, source from the row, guard returns."""
    source, net_hash = _write_source(tmp_path, _arch())
    verify_launch_anchor_pin(expected_anchor_sha256=net_hash, checkpoint_path=source,
                             trainer_step=0, run_id="run6")


def test_a_SWAPPED_artifact_at_the_pinned_path_REFUSES(tmp_path: Path) -> None:
    """The planted break, and the reason the pin is worth arming at all: same path, different
    weights. Nothing else in the launch notices — the run would train from a net nobody
    pre-registered and report a healthy boot."""
    import dataclasses

    _source, net_hash = _write_source(tmp_path, _arch())
    other, other_hash = _write_source(tmp_path / "swapped",
                                      dataclasses.replace(_arch(), hidden=64), run_id="other")
    assert other_hash != net_hash, "the swap fixture produced the same net; it proves nothing"
    with pytest.raises(RuntimeError, match="anchor sha256 mismatch"):
        verify_launch_anchor_pin(expected_anchor_sha256=net_hash, checkpoint_path=other,
                                 trainer_step=0, run_id="run6")


# ══ THE DENOMINATION EQUALITY, measured on a real artifact ═════════════════════════════
def test_the_row_hash_and_the_guard_hash_AGREE_on_the_artifact(tmp_path: Path) -> None:
    """`net_param_hash` over the stamp-rebuilt net vs `checkpoint_state_sha256` over the
    stored state. F-32 made them one denomination; this asserts they are one VALUE here."""
    source, net_hash = _write_source(tmp_path, _arch())
    assert checkpoint_state_sha256(source) == net_hash, (
        "the row's currency and the guard's currency disagree on this artifact, so a pin "
        "derived from the row would refuse the very file the row names"
    )


# ══ the SOURCE end: `init_trainer` gives the guard something to hash ═══════════════════
def test_init_trainer_sets_the_guards_fresh_init_source_from_the_row(tmp_path: Path) -> None:
    """THE SOURCE IS WRITTEN AT THE ARCH THE CONFIG RESOLVES, not at this file's narrow one:
    the warm start really fires here, and a source built at a different depth would be refused
    by `load_representation_policy_from_bc` on a key mismatch — a real guard, wrong subject."""
    from mantis.model.arch import arch_from_spec_and_config
    from mantis.train.orchestrator import init_trainer

    from _warmstart_config import minimal_config  # noqa: PLC0415

    base = minimal_config()
    base["identity"]["encoding"] = _ENC
    config_arch = arch_from_spec_and_config(lookup(_ENC), base)
    source, net_hash = _write_source(tmp_path, config_arch)
    trainer = init_trainer(config=_config_with_row(source, net_hash),
                           device=torch.device("cpu"), checkpoint_dir=tmp_path / "ckpt")
    # `BcWarmStart.checkpoint` is a `Path`, and `verify_launch_anchor_pin` takes `str | Path`,
    # so the comparison is made in the resolver's own type rather than by stringifying either.
    assert getattr(trainer, "checkpoint_source", None) == source, (
        "the fresh branch must name the warm-start artifact as the pin's verification source; "
        "without it an armed pin refuses every fresh launch"
    )


def test_init_trainer_leaves_the_source_None_when_no_row_is_declared(tmp_path: Path) -> None:
    """An absent row is the no-warm-start posture, and it must stay the no-pin posture too."""
    from _warmstart_config import minimal_config  # noqa: PLC0415

    from mantis.train.orchestrator import init_trainer

    cfg = minimal_config()
    cfg["identity"].pop("warm_start", None)
    trainer = init_trainer(config=cfg, device=torch.device("cpu"),
                           checkpoint_dir=tmp_path / "ckpt")
    assert getattr(trainer, "checkpoint_source", "unset") is None


# ══ THE MINTED ROW, on the artifact it names ═══════════════════════════════════════════
def test_the_minted_warm_start_row_names_an_artifact_whose_hashes_AGREE() -> None:
    """R336(c)'s 4b act, asserted where the act happened: the row minted into
    `configs/run6.yaml` names a checkpoint, and BOTH currencies must answer the same thing
    about it — otherwise the pin derived from the row refuses the very file the row names.

    **LOUD-SKIPS RATHER THAN PASSING when the artifact is absent, and that is the point.**
    `checkpoints/` is never tracked (R7), so the file exists on the box that produced it and
    nowhere else. A row that quietly passed on a missing file would be the phantom-gate shape:
    green everywhere, checked nowhere. The skip names what was not verified.
    """
    from mantis.config.loader import load_config
    from mantis.model import build_net
    from mantis.train.checkpoints import load_checkpoint

    repo = Path(__file__).resolve().parents[2]
    config_path = repo / "configs" / "run6.yaml"
    if not config_path.exists():
        pytest.skip("configs/run6.yaml is not in this tree — nothing minted, nothing to check")
    row = load_config(config_path).identity.warm_start
    assert row is not None, (
        "configs/run6.yaml carries no `identity.warm_start`: the 4b pin act's row is missing, "
        "and the launch pin it derives from would be None on the run that needs it"
    )
    artifact = repo / row.checkpoint
    if not artifact.is_file():
        pytest.skip(
            f"LOUD SKIP — the minted warm-start artifact {row.checkpoint} is not in this tree. "
            "`checkpoints/` is untracked by R7, so this row VERIFIES only where the checkpoint "
            f"lives. NOT verified here: net_param_hash == {row.net_hash[:12]}… and "
            "checkpoint_state_sha256 agreeing on it."
        )
    checkpoint = load_checkpoint(artifact)
    net = build_net(checkpoint.metadata.arch)
    net.load_state_dict(checkpoint.model_state)
    assert net_param_hash(net) == row.net_hash, (
        f"{row.checkpoint} rebuilds from its own stamp to net_param_hash "
        f"{net_param_hash(net)}, but the minted row declares {row.net_hash}. The artifact at "
        "that path is not the one run6 was pre-registered against"
    )
    assert checkpoint_state_sha256(artifact) == row.net_hash, (
        "the row's currency and the launch guard's currency disagree on the MINTED artifact: "
        f"{checkpoint_state_sha256(artifact)} vs {row.net_hash}. F-32's pin derives from the "
        "row, so this inequality would refuse every launch"
    )
