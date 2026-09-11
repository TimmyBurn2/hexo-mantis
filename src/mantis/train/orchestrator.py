"""Resume-precedence layer + fresh/resume trainer dispatch.

The frozen `RESUME_CHECKPOINT_OWNED_KEYS` set and the launch-wins override builder are pure
config-dict functions; `init_trainer` lives here too but LAZILY imports `Trainer`, so there is
no top-level `orchestrator → trainer` edge. What the frozen set reconciles is the LEGACY flat
training-config shape — on the new side arch/optimizer/scheduler ownership is STRUCTURAL and
encoding ownership is the baked `identity.encoding`.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Keys that MUST come from the CHECKPOINT on resume; the launch config wins for every other key.
# This is the LEGACY flat shape's set (pinned by T-CK-15); the nested shape's is the PATHS below.
RESUME_CHECKPOINT_OWNED_KEYS: frozenset[str] = frozenset({
    # encoding pins
    "encoding", "cluster_window_size", "cluster_threshold", "legal_move_radius", "board_size",
    # model architecture
    "in_channels", "input_channels", "res_blocks", "filters", "se_reduction_ratio", "model",
    # optimizer / scheduler / step state
    "total_steps", "scheduler_t_max", "eta_min", "min_lr", "lr", "weight_decay", "lr_schedule",
})

#: The same ownership on the nested `RunConfig` shape production resumes with: the identity the
#: stamp carries and the optimizer/scheduler values the restored state carries.
RESUME_CHECKPOINT_OWNED_PATHS: frozenset[str] = frozenset({
    "identity.encoding", "identity.representation", "identity.arch_kind",
    "train.lr", "train.weight_decay", "train.lr_schedule",
    "train.total_steps", "train.scheduler_t_max", "train.eta_min",
})

#: The override key carrying the owned launch values the builder dropped, so the resume can say
#: LOUDLY which declared values it ignored. A directive: stripped before the config is carried.
RESUME_OWNED_LAUNCH_VALUES_KEY = "resume_owned_launch_values"


def _split_owned_section(section: str, value: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (kept leaves, dropped owned leaves keyed by dotted path) for one nested section."""
    kept: dict[str, Any] = {}
    dropped: dict[str, Any] = {}
    for leaf, leaf_val in value.items():
        path = f"{section}.{leaf}"
        if path in RESUME_CHECKPOINT_OWNED_PATHS:
            dropped[path] = leaf_val
        else:
            kept[leaf] = leaf_val
    return kept, dropped


def _horizon_value(launch_config: Mapping[str, Any], key: str) -> Any:
    """The scheduler-horizon key off either shape: flat `key`, else nested `train.key`."""
    if launch_config.get(key) is not None:
        return launch_config[key]
    train = launch_config.get("train")
    return train.get(key) if isinstance(train, Mapping) else None


def build_resume_config_overrides(
    baked_config: Mapping[str, Any],
    launch_config: Mapping[str, Any],
    *,
    override_scheduler_horizon: bool = False,
    allow_fresh_scheduler: bool = False,
    declared_keys: frozenset | set | None = None,
) -> dict[str, Any]:
    """Build the resume `config_overrides` so the launch variant WINS.

    Seeded from `launch_config` minus `RESUME_CHECKPOINT_OWNED_KEYS`.
    `total_steps`/`scheduler_t_max` re-enter — re-horizoning the LR scheduler on load — ONLY
    under `--override-scheduler-horizon`. A `None` the operator EXPLICITLY declared travels; a
    `None` merely inherited is SKIPPED, so a stray null cannot nuke a real checkpoint value.
    """
    declared: frozenset = frozenset(declared_keys or ())
    overrides: dict[str, Any] = {}
    owned_launch: dict[str, Any] = {}
    for key, val in launch_config.items():
        if key in RESUME_CHECKPOINT_OWNED_KEYS:
            continue
        if isinstance(val, Mapping):
            # A nested section: its owned leaves stay with the checkpoint, the rest travel and
            # are merged LEAF-WISE onto the baked section by `apply_config_overrides_f1`.
            kept, dropped = _split_owned_section(key, val)
            owned_launch.update(dropped)
            overrides[key] = kept
        elif val is not None or key in declared:
            overrides[key] = val
    if owned_launch:
        overrides[RESUME_OWNED_LAUNCH_VALUES_KEY] = owned_launch
    # No `torch_compile[_mode]` injection: a LEGACY key with no consumer poisoned the carried
    # config, and write-time validation correctly rejected the first post-resume save.
    # Scheduler-horizon gate: only --override-scheduler-horizon re-horizons the LR anneal.
    if override_scheduler_horizon:
        for horizon in ("total_steps", "scheduler_t_max"):
            value = _horizon_value(launch_config, horizon)
            if value is not None:
                overrides[horizon] = int(value)
    if allow_fresh_scheduler:
        overrides["allow_fresh_scheduler"] = True
    return overrides


def init_trainer(
    *,
    config: Mapping[str, Any],
    device: Any,
    checkpoint_path: str | None = None,
    checkpoint_dir: Any = None,
    override_scheduler_horizon: bool = False,
    allow_fresh_scheduler: bool = False,
    declared_keys: frozenset | set | None = None,
    sink: Any = None,
) -> Any:
    """Fresh-run vs resume dispatch, thin against the typed config + `build_net(arch)`.

    `Trainer` is imported lazily inside the body, so there is no top-level
    `orchestrator → trainer` edge. `device` is REQUIRED and keyword-only with NO default:
    `Trainer.__init__` turns a `None` into CPU, so a caller that omitted it trained on CPU
    silently. The device is a config fact, and omitting it must be a `TypeError` at the call.
    """
    from mantis.train.trainer.core import Trainer  # lazy (Slice 2) — no top-level edge.

    if checkpoint_path is not None:
        from mantis.train.checkpoints import resume_trainer

        overrides = build_resume_config_overrides(
            config, config,
            override_scheduler_horizon=override_scheduler_horizon,
            allow_fresh_scheduler=allow_fresh_scheduler,
            declared_keys=declared_keys,
        )
        return resume_trainer(
            Trainer, checkpoint_path,
            fallback_config=config, config_overrides=overrides,
            declared_keys=declared_keys, sink=sink, device=device,
        )

    from mantis.encoding import resolve_from_config
    from mantis.model import arch_from_spec_and_config, build_net

    # `resolve_from_config` reads the nested `identity.encoding` shape as well as the legacy flat
    # one, so this site carries no copy of that knowledge.
    cfg = dict(config)
    spec = resolve_from_config(cfg)
    arch = arch_from_spec_and_config(spec, cfg)
    model = build_net(arch)

    # THE BC WARM-START ENTRY, on the FRESH branch only: a resume already restored trained
    # weights and seeding over them would destroy them. An absent row makes this a no-op.
    from mantis.train.warmstart import maybe_warmstart_gnn_from_bc, resolve_bc_warm_start

    maybe_warmstart_gnn_from_bc(model, cfg, spec=spec)

    # Pass the DECLARED arch (the SOLE arch source at save) + the injected sink through so a
    # fresh-run Trainer stamps envelope-v2 checkpoints from `metadata.arch` and routes events.
    trainer = Trainer(model, dict(config), arch=arch, checkpoint_dir=checkpoint_dir,
                      device=device, sink=sink)

    # The fresh branch's anchor pin source: `verify_launch_anchor_pin` FAILS CLOSED when a pin is
    # set and no `checkpoint_source` is readable, and the step-0 anchor IS the warm-start
    # artifact. Reading the row through the same resolver is a second READ, not a second authority.
    declared = resolve_bc_warm_start(cfg)
    trainer.checkpoint_source = None if declared is None else declared.checkpoint
    return trainer
