"""The launch pin's TWO ends, against the same warm-start row.

An inert guard and a guard that refuses every launch are both broken, and both ship green if
only one end is tested: the PIN end derives `expected_anchor_sha256` from `identity.warm_start`,
and the SOURCE end has `init_trainer`'s fresh branch set `checkpoint_source` from that same row.
The guard FAILS CLOSED when a pin is set and no source is readable.

The equality between the two currencies is MEASURED, not assumed: `warm_start.net_hash` hashes a
net REBUILT from the artifact's stamp, while the guard hashes the STORED state, so a buffer the
rebuild adds or the save drops moves one side and not the other.
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
    """A deliberately NARROW graph arch, off the dataclass defaults so a rebuild that ignored the stamp is visible."""
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


def test_a_pinned_fresh_init_REFUSES_when_no_source_is_readable() -> None:
    """What an armed pin did on EVERY fresh launch before `checkpoint_source` had a producer."""
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


def test_the_row_hash_and_the_guard_hash_AGREE_on_the_artifact(tmp_path: Path) -> None:
    """One denomination is not one VALUE: the stamp-rebuilt hash and the stored-state hash must
    agree on a real artifact."""
    source, net_hash = _write_source(tmp_path, _arch())
    assert checkpoint_state_sha256(source) == net_hash, (
        "the row's currency and the guard's currency disagree on this artifact, so a pin "
        "derived from the row would refuse the very file the row names"
    )


def test_init_trainer_sets_the_guards_fresh_init_source_from_the_row(tmp_path: Path) -> None:
    """The source is written at the arch the CONFIG resolves, not this file's narrow one: the warm
    start really fires here, and another depth would be refused on a key mismatch — wrong subject."""
    from mantis.model.arch import arch_from_spec_and_config
    from mantis.train.orchestrator import init_trainer

    from _warmstart_config import minimal_config  # noqa: PLC0415

    base = minimal_config()
    base["identity"]["encoding"] = _ENC
    config_arch = arch_from_spec_and_config(lookup(_ENC), base)
    source, net_hash = _write_source(tmp_path, config_arch)
    trainer = init_trainer(config=_config_with_row(source, net_hash),
                           device=torch.device("cpu"), checkpoint_dir=tmp_path / "ckpt")
    # Compared in the resolver's own `Path` type rather than by stringifying either side.
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


def test_the_minted_warm_start_row_names_an_artifact_whose_hashes_AGREE() -> None:
    """Both currencies must answer the same thing about the artifact the minted row names.

    LOUD-SKIPS rather than passing when the artifact is absent: `checkpoints/` is never tracked,
    so a quiet pass on a missing file would be green everywhere and checked nowhere.
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
