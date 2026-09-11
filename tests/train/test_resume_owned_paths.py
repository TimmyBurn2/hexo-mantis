"""The nested resume shape owns what the checkpoint carries: the flat legacy set filtered
nothing on a nested launch, so every launch section silently overrode the baked one."""
from __future__ import annotations

from pathlib import Path

import pytest

from mantis.config.loader import load_config
from mantis.config.schema import RunConfig
from mantis.train import orchestrator
from mantis.train.checkpoints import (
    RESUME_DIRECTIVE_KEYS,
    apply_config_overrides_f1,
    resume_trainer,
    save_checkpoint,
)
from mantis.train.orchestrator import (
    RESUME_CHECKPOINT_OWNED_KEYS,
    RESUME_CHECKPOINT_OWNED_PATHS,
    RESUME_OWNED_LAUNCH_VALUES_KEY,
    build_resume_config_overrides,
)
from mantis.train.trainer.core import Trainer

_REPO = Path(__file__).resolve().parents[2]


def _nested(**train: object) -> dict:
    """The minted dev config's dump with `train.*` leaves replaced — a real nested launch."""
    cfg = load_config(_REPO / "configs" / "dev_example.yaml").model_dump()
    cfg["train"].update(train)
    return cfg


def test_every_nested_owned_path_names_a_real_leaf_and_mirrors_the_flat_set() -> None:
    """Every dotted path names a schema leaf; every `train.*` leaf is in the flat set too."""
    fields = RunConfig.model_fields
    for path in RESUME_CHECKPOINT_OWNED_PATHS:
        section, leaf = path.split(".")
        section_model = fields[section].annotation
        assert section_model is not None and leaf in section_model.model_fields, (
            f"{path} names no schema leaf")
        if section == "train":
            assert leaf in RESUME_CHECKPOINT_OWNED_KEYS, f"{path} is owned nested but not flat"


def test_the_builder_drops_owned_leaves_and_records_them(monkeypatch) -> None:
    launch = _nested(eta_min=0.123, grad_clip=7.0)
    overrides = build_resume_config_overrides(launch, launch)
    assert "eta_min" not in overrides["train"], "an owned leaf must not travel as an override"
    assert overrides["train"]["grad_clip"] == 7.0, "a non-owned leaf still travels"
    assert "encoding" not in overrides["identity"]
    assert overrides[RESUME_OWNED_LAUNCH_VALUES_KEY]["train.eta_min"] == 0.123
    assert RESUME_OWNED_LAUNCH_VALUES_KEY in RESUME_DIRECTIVE_KEYS, "the record must be stripped"

    # Mutation self-test: with the nested set emptied the leaf travels, so the filter bites.
    monkeypatch.setattr(orchestrator, "RESUME_CHECKPOINT_OWNED_PATHS", frozenset())
    assert build_resume_config_overrides(launch, launch)["train"]["eta_min"] == 0.123


def test_the_leafwise_merge_keeps_baked_owned_leaves_and_replaces_the_rest() -> None:
    baked = {"train": {"lr": 1e-3, "eta_min": 5e-4, "grad_clip": 1.0, "ema": {"enabled": False}}}
    overrides = {"train": {"grad_clip": 2.0, "ema": {"enabled": True}}}
    resolved, deferred = apply_config_overrides_f1(baked, overrides, None)
    assert resolved["train"] == {"lr": 1e-3, "eta_min": 5e-4, "grad_clip": 2.0,
                                 "ema": {"enabled": True}}
    assert deferred == frozenset()
    # The declared/defer rule per leaf, dotted: a non-declared differing leaf defers by path.
    resolved, deferred = apply_config_overrides_f1(baked, {"train": {"grad_clip": 3.0}},
                                                   frozenset())
    assert resolved["train"]["grad_clip"] == 1.0 and deferred == frozenset({"train.grad_clip"})
    resolved, _ = apply_config_overrides_f1(baked, {"train": {"grad_clip": 3.0}},
                                            frozenset({"train.grad_clip"}))
    assert resolved["train"]["grad_clip"] == 3.0


@pytest.fixture
def spy_sink():
    class _Sink:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def emit(self, event: dict) -> None:
            self.events.append(dict(event))

        def named(self, name: str) -> list[dict]:
            return [e for e in self.events if e.get("event") == name]

    return _Sink()


def test_a_launch_eta_min_never_reaches_a_resumed_run_and_is_said_out_loud(
    tmp_path, tiny_net, optim_scaler_sched, metadata_kwargs, spy_sink,
) -> None:
    """LAW-07 producer: the baked owned leaf wins, the ignore is loud, the record is stripped."""
    opt, scaler, sched = optim_scaler_sched
    baked_cfg = _nested(eta_min=5e-4, lr_schedule="none")
    path = save_checkpoint(model=tiny_net, optimizer=opt, scaler=scaler, scheduler=sched,
                           step=100, config=baked_cfg, metadata_kwargs=metadata_kwargs,
                           checkpoint_dir=tmp_path, kind="full")
    launch = _nested(eta_min=0.0123, lr_schedule="none", grad_clip=7.0)
    overrides = build_resume_config_overrides(launch, launch)
    trainer = resume_trainer(Trainer, path, config_overrides=overrides, sink=spy_sink)

    assert trainer.config["train"]["eta_min"] == 5e-4, "the baked owned leaf must win on resume"
    assert trainer.config["train"]["grad_clip"] == 7.0, "the launch's non-owned leaf must win"
    assert RESUME_OWNED_LAUNCH_VALUES_KEY not in trainer.config, "the directive must be stripped"
    RunConfig.model_validate(trainer.config)
    ignored = spy_sink.named("resume_owned_launch_value_ignored")
    assert [e["knob"] for e in ignored] == ["train.eta_min"], ignored
    assert ignored[0]["declared"] == 0.0123 and ignored[0]["baked"] == 5e-4
