"""A full checkpoint loads STRICT, and a resume may not change identity.

Leniency is correct only for a bare anchor, whose key set is a genuine SUBSET of
`build_net`'s; on a FULL checkpoint a missing key means the stamped arch and the rebuilt
net disagree, and the run trains partly-random weights while its step counter says it is
continuing. Identity is compared against the RUN about to resume, not the artifact
against itself, which is what `load_checkpoint`'s encoding check already covers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

import _microbatch_harness as H
from mantis.train.checkpoints import (
    ResumeIdentityMismatchError,
    ResumeTargetSemanticsError,
    checkpoint_filename,
    content_sha8,
    load_checkpoint,
    resume_trainer,
)
from mantis.train.trainer.core import Trainer


def _rewrite(payload: dict, directory: Path) -> Path:
    """Re-publish a modified payload under a filename its own content hash validates.

    `_verify_provenance` rejects a mismatched name BEFORE the state dict is loaded, which
    would mask the strictness refusal this suite is about.
    """
    md = payload["metadata"]
    path = directory / checkpoint_filename(md["run_id"], int(md["step"]), content_sha8(payload))
    torch.save(payload, path)
    return path


def _write_full(tmp_path: Path) -> Path:
    trainer = H.tiny_graph_trainer(tmp_path, sink=H.SpySink())
    return trainer.save_checkpoint(None)


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
    """A weights-only artifact is a SUBSET by construction, so leniency stays keyed on kind."""
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


# These target-semantics leaves build no net, so every identity check above passes them;
# what they decide is whether a stored replay row is a visit-count distribution or a
# completed improved policy, and a restored ring carries no per-row provenance to tell
# the two apart. `search.kind` decides it, `train.policy_target` is what the stamp carries.
@pytest.mark.parametrize("section,leaf,value", [
    ("train", "policy_target", "completed_improved_policy"),
    ("search", "kind", "gumbel"),
])
def test_a_resume_that_moves_a_target_semantics_key_halts(
    tmp_path: Path, section: str, leaf: str, value: Any
) -> None:
    """Each leaf separately, so one parametrised row cannot pass by covering another."""
    path = _write_full(tmp_path)
    baked = load_checkpoint(path).config
    block = dict(baked[section])
    assert block.get(leaf) != value, "the fixture no longer moves the leaf it names"
    block[leaf] = value

    with pytest.raises(ResumeTargetSemanticsError, match=leaf):
        resume_trainer(
            Trainer, path, device=torch.device("cpu"),
            config_overrides={section: block},
            declared_keys=frozenset({section}),
        )


def test_a_resume_that_leaves_the_target_semantics_alone_proceeds(tmp_path: Path) -> None:
    """Mutation half: a guard that fires on every resume blocks every resume."""
    path = _write_full(tmp_path)
    baked = load_checkpoint(path).config
    trainer = resume_trainer(
        Trainer, path, device=torch.device("cpu"),
        config_overrides={"train": dict(baked["train"])},
        declared_keys=frozenset({"train"}),
    )
    assert trainer.loaded_from_full_checkpoint


def test_the_target_semantics_halt_names_both_sides(tmp_path: Path) -> None:
    """An operator reading the halt must not have to diff two files to act on it."""
    path = _write_full(tmp_path)
    train_block = dict(load_checkpoint(path).config["train"])
    train_block["policy_target"] = "completed_improved_policy"
    with pytest.raises(ResumeTargetSemanticsError) as excinfo:
        resume_trainer(
            Trainer, path, device=torch.device("cpu"),
            config_overrides={"train": train_block},
            declared_keys=frozenset({"train"}),
        )
    message = str(excinfo.value)
    assert "completed_improved_policy" in message and "raw_visit_distribution" in message, (
        f"the halt does not name the effective value and the checkpoint's: {message}"
    )


def test_the_search_regime_knobs_are_deliberately_not_target_semantics_keys(
    tmp_path: Path,
) -> None:
    """The considered OMISSION: `gumbel_m` changes a target's QUALITY, not its meaning,
    which puts it with the other unguarded search knobs a run legitimately varies."""
    path = _write_full(tmp_path)
    selfplay = dict(load_checkpoint(path).config["selfplay"])
    selfplay["gumbel_m"] = selfplay["gumbel_m"] + 8
    trainer = resume_trainer(
        Trainer, path, device=torch.device("cpu"),
        config_overrides={"selfplay": selfplay},
        declared_keys=frozenset({"selfplay"}),
    )
    assert trainer.loaded_from_full_checkpoint, (
        "a candidate-count change must NOT halt a resume — if this starts failing, the "
        "guard has widened past what it can justify"
    )


