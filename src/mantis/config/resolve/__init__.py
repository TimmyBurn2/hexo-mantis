"""Resolver family — one module per regime knob (mantis.config.resolve.*).

Each rule is the single authority both the self-play and eval callers import when they land, so
train/eval cannot re-derive a knob divergently (CONTEXT bug-class #3). None of these modules pull
torch (config → encoding, util only — DAG).
"""
from mantis.config.resolve.actor_sync import resolve_actor_sync_cadence
from mantis.config.resolve.amp import resolve_amp_dtype
from mantis.config.resolve.bootstrap import (
    BootstrapNotFoundError,
    ResolvedBootstrap,
    resolve_bootstrap,
)
from mantis.config.resolve.composition import (
    UnvalidatedConfigError,
    require_run_config,
)
from mantis.config.resolve.draw_rate import (
    DrawRateAbortSpec,
    resolve_draw_rate_abort,
)
from mantis.config.resolve.encoding import (
    UNSPECIFIED,
    AbsentEncodingError,
    EncodingConflictError,
    EncodingResolution,
    normalize_declared,
    normalize_stamp,
    reconcile_encoding,
)
from mantis.config.resolve.monitor import resolve_monitor_config
from mantis.config.resolve.nsims import resolve_eval_model_sims
from mantis.config.resolve.run_length import resolve_max_train_steps

__all__ = [
    "UNSPECIFIED",
    "AbsentEncodingError",
    "BootstrapNotFoundError",
    "DrawRateAbortSpec",
    "EncodingConflictError",
    "EncodingResolution",
    "ResolvedBootstrap",
    "UnvalidatedConfigError",
    "normalize_declared",
    "normalize_stamp",
    "reconcile_encoding",
    "require_run_config",
    "resolve_actor_sync_cadence",
    "resolve_amp_dtype",
    "resolve_bootstrap",
    "resolve_draw_rate_abort",
    "resolve_eval_model_sims",
    "resolve_max_train_steps",
    "resolve_monitor_config",
]
