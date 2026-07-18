"""Run-config schema (contract run-config-schema v1).

Every model is strict: unknown key = hard error, missing key = hard error, silent scalar
coercions (str->int, float->int, bool->int) rejected, values immutable. NO code-side
defaults — a default lives in exactly one place: the schema field (repo_design §5). Identity
keys carry no terminal defaults at all; representation is the closed set {grid, graph}
(registry.toml + repo_design §3 ground truth — LAW-11).
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mantis.encoding import EncodingRegistryError, lookup

SCHEMA_VERSION = 1


class StrictModel(BaseModel):
    """Base for every config model: unknown key = hard error, no coercion, immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class IdentityConfig(StrictModel):
    """Identity keys have no terminal defaults (repo_design §5): absent = error.

    ``representation`` is cross-checked against the encoding's registry representation at
    validation time (F1 runtime guard): a graph encoding declared ``representation: grid`` (or
    vice versa) is REJECTED at load, so the LAW-06 amp pin (resolve_amp_dtype reads this field)
    cannot be bypassed by a LAW-11-inconsistent config. Frozen sourced representation from the
    encoding spec, making disagreement structurally impossible; this guard restores that invariant.
    """

    encoding: str = Field(min_length=1)
    representation: Literal["grid", "graph"]

    @model_validator(mode="after")
    def _representation_matches_registry(self) -> "IdentityConfig":
        try:
            spec = lookup(self.encoding)
        except EncodingRegistryError as exc:
            raise ValueError(str(exc)) from exc
        if self.representation != spec.representation:
            raise ValueError(
                f"identity.representation={self.representation!r} disagrees with the registry "
                f"representation {spec.representation!r} for encoding {self.encoding!r} "
                "(LAW-11 identity consistency; a mismatch would bypass the LAW-06 amp-dtype pin)."
            )
        return self


class RadiusStage(StrictModel):
    """One (step, radius) point of a legal-move radius curriculum schedule."""

    step: int
    radius: int


class EvalConfig(StrictModel):
    """Eval opponent simulation counts (resolve_eval_model_sims reads these — no code default)."""

    random_model_sims: int
    sealbot_model_sims: int


class SelfplayConfig(StrictModel):
    """Self-play knobs in the WP8 field set.

    ``legal_move_radius_schedule`` is REQUIRED with no ``= None`` default: every config MUST
    write it explicitly (``null`` = "no curriculum → use the encoding's registry radius", an
    explicit-complete declaration, not a code-side default). resolve_radius_from_schedule reads it.
    """

    legal_move_radius_schedule: list[RadiusStage] | None


class RunConfig(StrictModel):
    """Top-level run config: explicit, complete, schema_version-pinned."""

    schema_version: int
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_\-]*$")
    seed: int
    identity: IdentityConfig
    eval: EvalConfig
    selfplay: SelfplayConfig

    @field_validator("schema_version")
    @classmethod
    def _pin_schema_version(cls, v: int) -> int:
        if v != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}, got {v}")
        return v
