"""Run-config schema (contract run-config-schema v1).

Every model is strict: unknown key = hard error, missing key = hard error, values
immutable. NO code-side defaults — a default lives in exactly one place: the schema
field (repo_design §5). Identity keys carry no terminal defaults at all.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = 1


class StrictModel(BaseModel):
    """Base for every config model: unknown key = hard error, values immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class IdentityConfig(StrictModel):
    """Identity keys have no terminal defaults (repo_design §5): absent = error."""

    encoding: str = Field(min_length=1)
    representation: Literal["dense", "graph"]


class RunConfig(StrictModel):
    """Top-level run config: explicit, complete, schema_version-pinned."""

    schema_version: int
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_\-]*$")
    seed: int
    identity: IdentityConfig

    @field_validator("schema_version")
    @classmethod
    def _pin_schema_version(cls, v: int) -> int:
        if v != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}, got {v}")
        return v
