"""Resume-precedence layer + fresh/resume trainer dispatch (WP10 §a.4/§c.2).

The PURE loader-adjacent config-dict functions land in Slice 1 (T-CK-14/15/17 import
`RESUME_CHECKPOINT_OWNED_KEYS` + `build_resume_config_overrides` from here): the frozen
`RESUME_CHECKPOINT_OWNED_KEYS` set, the launch-wins override builder, and the
operator-declaration classifier. `init_trainer` (fresh-vs-resume dispatch) also lives here
but LAZILY imports `Trainer` (Slice 2) inside its body — no top-level `orchestrator → trainer`
edge, so this module imports clean at Slice 1.

What the frozen set + reconciler reconcile is the LEGACY flat training-config shape (a
pre-v2/migration-resume concern): on the pure new side arch/optimizer/scheduler ownership
is STRUCTURAL (build from `metadata.arch`; restore from state) and encoding-ownership is
the baked `identity.encoding`. Its LAW-08 live consumer is the conformance suite
(T-CK-14..20) + the legacy-resume path.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# ── The checkpoint-owned frozen key set — SINGLE source of truth (repo_design §c.2) ────
# Keys that MUST come from the CHECKPOINT on resume, never from the launch variant; the
# launch config wins for every OTHER key. Imported by T-CK-15 (mutating add/remove bites).
RESUME_CHECKPOINT_OWNED_KEYS: frozenset[str] = frozenset({
    # encoding pins
    "encoding", "cluster_window_size", "cluster_threshold", "legal_move_radius", "board_size",
    # model architecture
    "in_channels", "input_channels", "res_blocks", "filters", "se_reduction_ratio", "model",
    # optimizer / scheduler / step state
    "total_steps", "scheduler_t_max", "eta_min", "min_lr", "lr", "weight_decay", "lr_schedule",
})


def compute_declared_keys(layers: list[dict] | None) -> frozenset[str]:
    """Top-level keys the OPERATOR declared in a `config`/`variant` layer (CONFRES F1/B3).

    A key present in a `config`/`variant` layer is a DECLARATION (incl. an explicit
    `key: null`); a key present only in a `base` layer is INHERITED and DEFERS to a
    checkpoint-baked value on resume. `None`/empty → empty set.
    """
    if not layers:
        return frozenset()
    declared: set[str] = set()
    for layer in layers:
        if isinstance(layer, dict) and layer.get("kind") in ("config", "variant"):
            raw = layer.get("raw") or {}
            if isinstance(raw, dict):
                declared.update(raw.keys())
    return frozenset(declared)


def build_resume_config_overrides(
    baked_config: Mapping[str, Any],
    launch_config: Mapping[str, Any],
    *,
    override_scheduler_horizon: bool = False,
    allow_fresh_scheduler: bool = False,
    declared_keys: frozenset | set | None = None,
) -> dict[str, Any]:
    """Build the resume `config_overrides` so the launch variant WINS (D-FULLSPEC E0).

    Seeds the overrides from `launch_config` (operator intent) minus
    `RESUME_CHECKPOINT_OWNED_KEYS` (encoding/arch pins + optimizer/scheduler/step state). The
    `--override-scheduler-horizon` gate is preserved verbatim: `total_steps`/`scheduler_t_max`
    re-enter the overrides (re-horizoning the LR scheduler on load) ONLY when the flag is set;
    without it the restored scheduler `T_max` is untouched. `baked_config` is accepted for the
    resume round-trip (the F1 defer against it runs in `apply_config_overrides_f1`); the
    override set itself is a function of `launch_config`.

    B3 null semantics: a `None` the operator EXPLICITLY declared travels; a `None` merely
    inherited is SKIPPED so a stray null cannot nuke a real checkpoint value.
    """
    declared: frozenset = frozenset(declared_keys or ())
    overrides: dict[str, Any] = {
        key: val
        for key, val in launch_config.items()
        if key not in RESUME_CHECKPOINT_OWNED_KEYS and (val is not None or key in declared)
    }
    # torch_compile[_mode] always travel with a concrete value on resume (pre-E0 default path).
    overrides["torch_compile"] = launch_config.get("torch_compile", False)
    if launch_config.get("torch_compile_mode") is not None:
        overrides["torch_compile_mode"] = launch_config["torch_compile_mode"]
    # Scheduler-horizon gate: only --override-scheduler-horizon re-horizons the LR anneal.
    if override_scheduler_horizon:
        if launch_config.get("total_steps") is not None:
            overrides["total_steps"] = int(launch_config["total_steps"])
        if launch_config.get("scheduler_t_max") is not None:
            overrides["scheduler_t_max"] = int(launch_config["scheduler_t_max"])
    if allow_fresh_scheduler:
        overrides["allow_fresh_scheduler"] = True
    return overrides


def init_trainer(
    *,
    config: Mapping[str, Any],
    checkpoint_path: str | None = None,
    checkpoint_dir: Any = None,
    device: Any = None,
    override_scheduler_horizon: bool = False,
    allow_fresh_scheduler: bool = False,
    declared_keys: frozenset | set | None = None,
    sink: Any = None,
) -> Any:
    """Fresh-run vs resume dispatch, rebuilt thin against the typed config + `build_net(arch)`.

    Lazily imports `Trainer` (Slice 2) inside the body so there is no top-level
    `orchestrator → trainer` import edge (the module imports clean at Slice 1; this function
    is exercised only at Slice 2 / O-SMOKE).
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

    # `resolve_from_config` reads the WP8 nested `identity.encoding` shape as well as the legacy
    # flat one (TD-4 / CARD-POOL-ENCODING-BRIDGE), so a fresh Trainer builds the DECLARED arch
    # without this site carrying its own copy of that knowledge.
    cfg = dict(config)
    spec = resolve_from_config(cfg)
    arch = arch_from_spec_and_config(spec, cfg)
    model = build_net(arch)
    # Pass the DECLARED arch (the SOLE arch source at save) + the injected sink through so a
    # fresh-run Trainer stamps envelope-v2 checkpoints from `metadata.arch` and routes events.
    return Trainer(model, dict(config), arch=arch, checkpoint_dir=checkpoint_dir,
                   device=device, sink=sink)
