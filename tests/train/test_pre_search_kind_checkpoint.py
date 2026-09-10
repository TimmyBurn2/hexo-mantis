"""⊕ what a PRE-`search.kind` checkpoint does against the live schema, stated outright.

THE CLASS, and it is not hypothetical. `load_checkpoint` schema-validates the checkpoint's
EMBEDDED config through the LIVE `RunConfig`. That makes every REQUIRED-with-no-default field
addition a break in every artifact written before it: the run6 BC warm-start could not be
loaded at HEAD because GUMBEL-REPAIR-1 added `selfplay.gumbel_variant` and
`gumbel_root_counts`, and nothing in the suite caught it — the schema grew, the tests stayed
green, and every warm start and every `--resume-from` from an older artifact was bricked.

WHAT THIS BRANCH DOES TO IT: WORSE, BY ONE ERROR CLASS, AND SAID SO HERE RATHER THAN LEFT TO
BE DISCOVERED. A pre-branch config now fails TWICE — it carries keys `extra="forbid"` rejects
(`selfplay.gumbel_mcts` and the two `completed_q_values`) AND lacks the one this branch
requires (`search.kind`). Giving `search.kind` a default is NOT the fix: R1 forbids a
code-side default and LAW-11 makes an absent identity key an error.

WHAT THIS FILE THEREFORE ASSERTS is the outcome as it stands, on both halves:

  1. the refusal HAPPENS, and names both halves, so the class is visible; and
  2. the SANCTIONED RECOVERY still works — `strip_and_restamp` (LAW-12's one weights-only
     path) never reads the embedded config, it re-synthesises one from the live schema, and
     the artifact it produces loads clean and carries `search.kind`.

(2) is the property the wave-3 re-mint actually depends on, and it is the reason this branch
does not strand the warm start even though it widens the refusal.

THE REAL DEFECT IS NOT FIXED HERE, and the reason is scope rather than difficulty: a
checkpoint's embedded config is a HISTORICAL RECORD, and validating a record against today's
schema asks the wrong question. Changing that means moving `docs/design/repo_design.md` §6's
*"schema-validated on write AND read"* and retiring T-CK-04
(`test_config_snapshot_schema_validated_on_read`) — a checkpoint-contract change (contract
#4, LAW-12 territory) that belongs to a ruling, not to the tail of a search packet. This file
is the witness that keeps the class from going quiet in the meantime: when the loader stops
validating records, arm (1) reds and its message says what to do.
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
    """The same config as it would have been written BEFORE this branch."""
    old = copy.deepcopy(config)
    old.pop("search", None)
    for section, leaf, value in _DELETED:
        old[section][leaf] = value
    return old


def _rewrite_with_config(path: Path, config: dict[str, Any], out_dir: Path) -> Path:
    """Re-save a saved envelope with a different embedded config, under its correct
    provenance filename — a stale content hash would otherwise mask the field under test."""
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
    """Arm 1 — the refusal, and what it says.

    It must name BOTH halves, because an operator reading only "extra key" would delete the
    old keys and hit the missing one on the next attempt.
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
    """Arm 2 — LAW-12's one path is unaffected, which is why the warm start is not stranded.

    `strip_and_restamp` reads the raw payload and re-synthesises a config from the LIVE
    schema; it never validates the embedded one. The artifact it writes loads clean.

    MUTATION THAT REDS IT: a strip whose synthetic config forgot `search.kind` — the write
    would fail schema validation, which is the same class one layer up.
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
    """The control: the refusal above is about the AGE of the config, not about this file's
    fixture. A config written by this branch loads."""
    opt, scaler, sched = optim_scaler_sched
    live = save_checkpoint(
        model=tiny_net, optimizer=opt, scaler=scaler, scheduler=sched, step=100,
        config=valid_config, metadata_kwargs=metadata_kwargs, checkpoint_dir=tmp_path,
        kind="full",
    )
    assert load_checkpoint(live).config["search"]["kind"] == "puct"
