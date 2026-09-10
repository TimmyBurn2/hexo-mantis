"""What a pre-`search.kind` checkpoint does against the live schema.

`load_checkpoint` schema-validates the checkpoint's EMBEDDED config through the live
`RunConfig`, so every required-with-no-default field addition breaks every older artifact.
Arm 1 pins that the refusal happens and names both halves (the keys `extra="forbid"` rejects
AND the missing `search.kind`); arm 2 pins that `strip_and_restamp` — which re-synthesises a
config from the live schema rather than reading the embedded one — still recovers the
artifact, which is what the wave-3 re-mint depends on.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from mantis.train.checkpoints import (
    checkpoint_filename,
    content_sha8,
    load_checkpoint,
    save_checkpoint,
    strip_and_restamp,
)

#: The four leaves this branch DELETED plus the one it ADDED — the exact difference between a
#: pre-branch embedded config and one the live schema accepts.
_DELETED = (
    ("selfplay", "gumbel_mcts", False),
    ("selfplay", "gumbel_variant", "legacy"),
    ("selfplay", "gumbel_root_counts", True),
    ("selfplay", "completed_q_values", False),
    ("train", "completed_q_values", False),
)


def _pre_branch(config: dict[str, Any]) -> dict[str, Any]:
    """Return the config as it would have been written before this branch."""
    old = copy.deepcopy(config)
    old.pop("search", None)
    for section, leaf, value in _DELETED:
        old[section][leaf] = value
    return old


def _rewrite_with_config(path: Path, config: dict[str, Any], out_dir: Path) -> Path:
    """Re-save an envelope with a different embedded config under its correct provenance filename.

    A stale content hash would otherwise mask the field under test.
    """
    payload = torch.load(path, weights_only=True)
    payload["config"] = config
    md = payload["metadata"]
    dst = out_dir / checkpoint_filename(md["run_id"], md["step"], content_sha8(payload))
    torch.save(payload, dst)
    return dst


@pytest.fixture
def pre_branch_checkpoint(tmp_path, tiny_net, optim_scaler_sched, valid_config,
                          metadata_kwargs) -> Path:
    opt, scaler, sched = optim_scaler_sched
    live = save_checkpoint(
        model=tiny_net, optimizer=opt, scaler=scaler, scheduler=sched, step=100,
        config=valid_config, metadata_kwargs=metadata_kwargs, checkpoint_dir=tmp_path,
        kind="full",
    )
    return _rewrite_with_config(live, _pre_branch(valid_config), tmp_path)


def test_a_pre_branch_checkpoint_is_REFUSED_and_names_both_halves(pre_branch_checkpoint):
    """Prove an old checkpoint is refused and the message names both the extra and missing keys.

    Naming only one half sends the operator round the loop twice.
    """
    with pytest.raises(ValidationError) as excinfo:
        load_checkpoint(pre_branch_checkpoint)
    message = str(excinfo.value)
    assert "search" in message, (
        f"the refusal does not name the MISSING key: {message}"
    )
    assert any(leaf in message for _s, leaf, _v in _DELETED), (
        f"the refusal does not name any of the DELETED keys it is rejecting: {message}"
    )


def test_the_sanctioned_weights_strip_still_recovers_it(pre_branch_checkpoint, tmp_path):
    """Prove the sanctioned weights-strip still recovers a refused checkpoint.

    `strip_and_restamp` re-synthesises the config from the live schema and never validates
    the embedded one, so the artifact it writes loads clean.
    """
    out = tmp_path / "stripped"
    out.mkdir()
    stripped = strip_and_restamp(
        pre_branch_checkpoint,
        new_encoding="gnn_axis_v1",
        run_id="recovered",
        checkpoint_dir=out,
        declared_encoding="gnn_axis_v1",
    )
    ck = load_checkpoint(stripped)
    assert ck.kind == "weights"
    assert ck.model_state, "the recovered artifact must still carry its weights"
    assert ck.config["search"]["kind"] in ("puct", "gumbel"), (
        "the re-synthesised config must declare a search kind — an artifact whose config "
        "cannot say which search produced it is the provenance gap this key closes"
    )
    for section, leaf, _value in _DELETED:
        assert leaf not in ck.config[section], (
            f"the re-synthesised config still carries the deleted {section}.{leaf}"
        )


def test_a_LIVE_checkpoint_round_trips(tmp_path, tiny_net, optim_scaler_sched, valid_config,
                                       metadata_kwargs):
    """Control: a config written at HEAD loads, so the refusal above is about the config's age."""
    opt, scaler, sched = optim_scaler_sched
    live = save_checkpoint(
        model=tiny_net, optimizer=opt, scaler=scaler, scheduler=sched, step=100,
        config=valid_config, metadata_kwargs=metadata_kwargs, checkpoint_dir=tmp_path,
        kind="full",
    )
    assert load_checkpoint(live).config["search"]["kind"] == "puct"
