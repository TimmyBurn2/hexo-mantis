"""`require_run_config` — the ONE composition-input rule (WPAX Phase S, DESIGN_S §1).

A composition root reads config SECTIONS (train, monitor, eval, identity). Duck-typing them
with `getattr(config, "<section>", None)` gave every absent section a silent default, and one
of those defaults — a bare `MonitorConfig()` — silently DISARMED the actor-lag hard abort that
`configs/run5.yaml` ships armed (ADJ-07). This module replaces the whole family with a single
gate: composition requires a schema-validated `RunConfig`, and after the gate every section is
a plain typed attribute read.

Why it lives HERE and not in `mantis.run`: `mantis.run` *is* a composition root, so putting the
input rule there would make it un-importable by any second composition surface (Phase P's
preflight is the first one coming) without importing the root itself. The rule is a
config-layer fact, so it lives in the config layer.
"""
from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from mantis.config.schema import RunConfig


class UnvalidatedConfigError(ValueError):
    """A composition root was handed something that is not a validated `RunConfig`.

    Follows the `mantis.config` error convention (`DuplicateKeyError`, `AbsentEncodingError`,
    `EncodingConflictError`, `BootstrapNotFoundError`): a named subclass of the nearest builtin,
    defined in the module that owns the fact, whose message names both the offending input and
    the operator's remedy.
    """


def require_run_config(config: Any, *, caller: str) -> RunConfig:
    """Return `config` iff it is a validated `RunConfig`; raise `UnvalidatedConfigError` else.

    The check reads the object's REAL type — `issubclass(type(config), RunConfig)` — not
    `isinstance`. `unittest.mock` sets `__class__` on a spec'd mock and `isinstance` honours it,
    so `isinstance(Mock(spec=RunConfig), RunConfig)` is True for an object that has been through
    no validation at all; `type()` cannot be spoofed that way. A genuine subclass still passes:
    it reached this root through `model_validate`, so every cross-field validator ran on it and
    rejecting it would reject a MORE validated object than the one we accept (LSP).

    KNOWN AND OUT OF SCOPE (WP-R §9.10) — the CLASS, not one method name. Pydantic offers
    several ways to build a real, genuinely-typed `RunConfig` that skipped the model-level
    validators: `RunConfig.model_construct(...)`, `cfg.model_copy(update=…)`, and any future
    sibling. **No type-based gate can see any of them** — they are not spoofs, they are the
    class. Two facts about that hole, and only one of them is reassuring:

      * `model_construct(...)` fails loudly and almost immediately: the sections it never
        set are simply absent, so `config.monitor` raises `AttributeError`. (Even this is
        imprecise for the `model_construct(**cfg.model_dump())` form, where the first typed
        read returns a `dict` and the failure lands one hop later.)
      * `model_copy(update=…)` does **NOT** fail at all. Every section is a real validated
        sub-model, every typed read succeeds — but the CROSS-FIELD validators never re-ran,
        so a copy can carry `actor_sync_cadence_steps >= max_train_steps` (the reachability
        bound), which the loader refuses and this root would drive: 20 steps, ONE sync, a
        frozen actor, with the lag threshold out of reach too (WPAX RED-TEAM F-3, measured).

    So the "it fails LOUD" argument holds for `model_construct` and for nothing else, and
    THIS function's guarantee must be read as exactly what it is: **the root is a
    `RunConfig`, which is not the same statement as "the root is a config the loader would
    accept."** That second statement is `revalidate_run_config`'s job, below, and a
    composition root needs BOTH — the type rule cannot be strengthened into the validity
    rule in place, because `require_run_config` is contractually IDENTITY-preserving and
    re-validation is not.

    Likewise SECTION shape: this function's subject is the ROOT object. A validated
    `RunConfig` carries every section, so the sections are typed for every object that
    reached it through `load_config` / `model_validate`; a wrong-shaped section smuggled
    onto a genuine `RunConfig` by one of the construction paths above is NOT caught here.
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

    The second half of the composition-input rule (WPAX RED-TEAM F-3). `require_run_config`
    answers "is this the class?"; this answers "is this a config the loader would accept?",
    and the two are different questions because pydantic can build a genuine, genuinely-typed
    `RunConfig` that never saw a model validator:

        base  = load_config("configs/smoke_gnn.yaml")
        train = base.train.model_copy(update={"max_train_steps": 20,
                                              "actor_sync_cadence_steps": 1000})
        bad   = base.model_copy(update={"train": train})

    `bad` is a real `RunConfig`, every typed read succeeds, and it drives a 20-step run with
    exactly ONE actor sync — run3's frozen actor — while `load_config` refuses the identical
    payload with "a cadence the run never reaches". Measured. `model_copy(update=…)` is also
    the idiomatic pydantic-v2 way to rig a config on a copy, so this is not a contrived route:
    it is the route a mutation corpus reaches for first.

    Re-validating the dump closes `model_copy(update=…)`, `model_construct(...)` and any
    post-gate mutation in ONE move, because it re-runs the cross-field validators (the F-2
    reachability bound among them) rather than enumerating construction paths — the
    enumeration this lineage has now been beaten by four times.

    `type(config)` rather than `RunConfig`: a validated SUBCLASS must survive this hop as
    itself, for the same LSP reason `require_run_config` admits it.

    NOT folded into `require_run_config`: that function is contractually identity-preserving
    (`assert require_run_config(x) is x` is pinned in a byte-frozen oracle, for the subclass
    row AND for the `model_construct` row that documents the hole), and re-validation returns
    a new object. Two rules, two functions, one call site each.
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
