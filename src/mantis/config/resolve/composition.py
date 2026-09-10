"""`require_run_config` — the ONE composition-input rule.

Duck-typing config sections with `getattr(config, "<section>", None)` gave every absent section
a silent default, and a bare `MonitorConfig()` once disarmed the actor-lag hard abort that
`configs/run5.yaml` ships armed. Composition requires a schema-validated `RunConfig`; after the
gate every section is a plain typed attribute read.

It lives in the config layer rather than in `mantis.run` because `mantis.run` IS a composition
root: a second composition surface would have to import the root to reach the rule.
"""
from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from mantis.config.schema import RunConfig


class UnvalidatedConfigError(ValueError):
    """A composition root was handed something that is not a validated `RunConfig`."""


def require_run_config(config: Any, *, caller: str) -> RunConfig:
    """Return `config` iff it is a validated `RunConfig`.

    The check reads the REAL type — `issubclass(type(config), RunConfig)` — because
    `unittest.mock` sets `__class__` on a spec'd mock and `isinstance` honours it.

    The guarantee is "the root is a `RunConfig`", NOT "a config the loader would accept":
    `model_construct` and `model_copy(update=…)` skip the model validators and no type-based
    gate can see them, which is `revalidate_run_config`'s half of the rule.

    Raises:
        UnvalidatedConfigError: `config` is not a `RunConfig`.
    """
    if not (isinstance(type(config), type) and issubclass(type(config), RunConfig)):
        raise UnvalidatedConfigError(
            f"{caller} requires a schema-validated mantis.config.schema.RunConfig; got "
            f"{type(config).__name__!r}.\n"
            "Every config section this root reads (train, monitor, eval, identity) arrives on "
            "the validated ROOT: a config object without them is rejected HERE rather than "
            "duck-typed into a smoke default (R1) or a silent disarm (ADJ-07). Smoke runs get "
            "smoke CONFIGS — mint one "
            "with tools/mint_config.py, or load an existing configs/*.yaml through "
            "mantis.config.loader.load_config."
        )
    return config


def revalidate_run_config(config: RunConfig, *, caller: str) -> RunConfig:
    """Re-run every model validator on `config` and return the VALIDATED result.

    `model_copy(update=…)` and `model_construct` build a real, genuinely-typed `RunConfig` whose
    CROSS-FIELD validators never ran, so a copy can carry `actor_sync_cadence_steps >=
    max_train_steps` (measured: a 20-step run with ONE actor sync) that `load_config` refuses.
    Re-validating the dump closes every such path at once instead of enumerating them.

    `type(config)` rather than `RunConfig` so a validated subclass survives the hop as itself;
    kept separate from the identity-preserving `require_run_config` because this returns a new
    object.

    Raises:
        UnvalidatedConfigError: `config` does not satisfy its own schema.
    """
    try:
        return type(config).model_validate(config.model_dump())
    except ValidationError as exc:
        raise UnvalidatedConfigError(
            f"{caller} was handed a mantis.config.schema.RunConfig that does NOT satisfy its "
            "own schema: it was built by a path that skips the model validators "
            "(`model_copy(update=…)`, `model_construct(...)`, or a post-validation mutation), "
            "so it is the right TYPE and an invalid CONFIG.\n"
            "Composition re-validates, because a config the loader would reject is a config "
            "the run must not be driven from — the two knobs below fail OPEN together "
            "(an unreachable sync cadence freezes the actor and an unreachable lag threshold "
            "hides it). Rig runs by re-loading a MINTED config with the values you want, not "
            f"by copying a loaded one.\n{exc}"
        ) from exc


__all__ = ["UnvalidatedConfigError", "require_run_config", "revalidate_run_config"]
