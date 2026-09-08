"""R345(b)(3) — a full checkpoint loads STRICT, and a resume may not change identity.

TWO REFUSALS THAT DID NOT EXIST.

STRICTNESS. `resume_trainer` loaded every checkpoint with `strict=False`, and the comment
explaining it is about a BARE ANCHOR: a weights-only artifact is a genuine SUBSET of the
`build_net` key set (T-CK-25), so strict would reject it spuriously. That reasoning is sound
and is preserved. What it never justified is applying the same leniency to a FULL checkpoint,
where a missing key means the arch that was stamped and the net that was rebuilt disagree —
and the run then trains a partly-randomly-initialised model while its optimizer state,
scheduler and step counter all say it is continuing. The one condition that silently discards
learned weights was the one condition nothing checked.

IDENTITY. `resume_trainer` APPLIES `config_overrides` onto the checkpoint's baked config
(CONFRES F1(A)), and nothing compares the result's identity block against the artifact. The
encoding half has a check — `load_checkpoint` refuses when `metadata.encoding_name` and
`config.identity.encoding` disagree — but that compares the checkpoint against ITSELF, not
against the run about to resume from it. A resume that changes `representation` or
`arch_kind` gets a net built from the checkpoint's stamped arch and a config claiming
another, which is LAW-11's subject one layer out.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

import _microbatch_harness as H
from mantis.train.checkpoints import (
    ResumeIdentityMismatchError,
    checkpoint_filename,
    content_sha8,
    load_checkpoint,
    resume_trainer,
)
from mantis.train.trainer.core import Trainer


def _rewrite(payload: dict, directory: Path) -> Path:
    """Re-publish a modified payload under a filename its own content hash validates.

    `_verify_provenance` re-derives `{run_id}_{step:08d}_{sha8}` from the payload and compares
    it against the filename, so a tampered payload saved under any other name is rejected for
    provenance BEFORE the state dict is ever loaded — the check would mask the one this suite
    is about. Naming it correctly is what makes the strictness assertion reachable.
    """
    md = payload["metadata"]
    path = directory / checkpoint_filename(md["run_id"], int(md["step"]), content_sha8(payload))
    torch.save(payload, path)
    return path


def _write_full(tmp_path: Path) -> Path:
    trainer = H.tiny_graph_trainer(tmp_path, sink=H.SpySink())
    return trainer.save_checkpoint(None)


# ── strictness ──────────────────────────────────────────────────────────────────────────
def test_a_full_checkpoint_missing_a_weight_is_refused(tmp_path: Path) -> None:
    """The defect: a key silently dropped, and the run continues on a random tensor."""
    path = _write_full(tmp_path)
    payload = torch.load(path, weights_only=True, map_location="cpu")
    assert payload["kind"] == "full"
    dropped = sorted(payload["model_state"])[0]
    del payload["model_state"][dropped]
    torn = _rewrite(payload, tmp_path)

    with pytest.raises(RuntimeError) as excinfo:
        resume_trainer(Trainer, torn, device=torch.device("cpu"))
    assert dropped in str(excinfo.value) or "Missing key" in str(excinfo.value), (
        f"the load did not name the missing key: {excinfo.value}"
    )


def test_a_full_checkpoint_with_an_unexpected_weight_is_refused(tmp_path: Path) -> None:
    """The mirror case: a key the arch does not have means the artifact is not this net."""
    path = _write_full(tmp_path)
    payload = torch.load(path, weights_only=True, map_location="cpu")
    payload["model_state"]["a_layer_that_does_not_exist.weight"] = torch.zeros(3)
    torn = _rewrite(payload, tmp_path)

    with pytest.raises(RuntimeError, match="a_layer_that_does_not_exist|Unexpected key"):
        resume_trainer(Trainer, torn, device=torch.device("cpu"))


def test_a_healthy_full_checkpoint_still_resumes(tmp_path: Path) -> None:
    """Mutation half: strictness that refuses everything is not strictness."""
    path = _write_full(tmp_path)
    trainer = resume_trainer(Trainer, path, device=torch.device("cpu"))
    assert trainer.loaded_from_full_checkpoint
    assert trainer.step == load_checkpoint(path).metadata.step


def test_a_bare_anchor_subset_still_loads_leniently(tmp_path: Path) -> None:
    """T-CK-25's reason is preserved: `kind == "weights"` is a SUBSET by construction.

    Strictness keyed on the kind rather than applied everywhere is the whole shape of this
    half — a rule that also rejected anchors would have made the leg a regression.
    """
    path = _write_full(tmp_path)
    payload = torch.load(path, weights_only=True, map_location="cpu")
    payload["kind"] = "weights"
    for key in ("optimizer_state", "scaler_state", "scheduler_state"):
        payload.pop(key, None)
    subset = sorted(payload["model_state"])
    del payload["model_state"][subset[0]]
    anchor = _rewrite(payload, tmp_path)

    trainer = resume_trainer(Trainer, anchor, device=torch.device("cpu"))
    assert not trainer.loaded_from_full_checkpoint, (
        "a weights artifact was read as a full resume"
    )


# ── identity ────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("key,value", [
    ("representation", "grid"),
    ("arch_kind", "GnnArchV2"),
    ("encoding", "gnn_axis_r8"),
])
def test_a_resume_that_moves_an_identity_key_halts(tmp_path: Path, key: str, value: Any) -> None:
    """Each identity leaf, separately: one parametrised row cannot pass by covering another."""
    path = _write_full(tmp_path)
    baked = load_checkpoint(path).config
    identity = dict(baked["identity"])
    assert identity.get(key) != value, "the fixture no longer moves the key it names"
    identity[key] = value

    with pytest.raises(ResumeIdentityMismatchError, match=key):
        resume_trainer(
            Trainer, path, device=torch.device("cpu"),
            config_overrides={"identity": identity},
            declared_keys=frozenset({"identity"}),
        )


def test_a_resume_that_leaves_identity_alone_proceeds(tmp_path: Path) -> None:
    """Mutation half: an identity check that fires on every resume blocks every resume."""
    path = _write_full(tmp_path)
    baked = load_checkpoint(path).config
    trainer = resume_trainer(
        Trainer, path, device=torch.device("cpu"),
        config_overrides={"identity": dict(baked["identity"])},
        declared_keys=frozenset({"identity"}),
    )
    assert trainer.loaded_from_full_checkpoint


def test_the_halt_names_both_sides(tmp_path: Path) -> None:
    """An operator reading the halt must not have to go and diff two files to act on it."""
    path = _write_full(tmp_path)
    identity = dict(load_checkpoint(path).config["identity"])
    identity["representation"] = "grid"
    with pytest.raises(ResumeIdentityMismatchError) as excinfo:
        resume_trainer(
            Trainer, path, device=torch.device("cpu"),
            config_overrides={"identity": identity},
            declared_keys=frozenset({"identity"}),
        )
    message = str(excinfo.value)
    assert "grid" in message and "graph" in message, (
        f"the halt does not name the effective value and the checkpoint's: {message}"
    )
