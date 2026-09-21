"""`ModelConfig` — the net's declared shape as config leaves (contract v35); `select_arch` read the trunk widths off two flat keys no schema carried, so every run built at the dataclass defaults."""
from pydantic import Field

from mantis.config.schema._base import StrictModel


class GnnWidthsConfig(StrictModel):
    """The graph trunk's shape, ONE block (the readout is `num_layers × hidden` wide): `ge=1` on both, no off value, graph-scoped by `ARCH_SCOPED_KEYS`."""

    hidden: int = Field(ge=1)
    num_layers: int = Field(ge=1)


class AuxSoftPolicyConfig(StrictModel):
    """The auxiliary soft-policy head's rows (v36): `target_temperature` (`gt=1`: at 1 the head reads armed while dead) over the target's explicit entries, `weight` (`gt=0`) scaling its CE into the step's loss."""

    target_temperature: float = Field(gt=1.0)
    weight: float = Field(gt=0.0)


class ModelConfig(StrictModel):
    """The net's declared shape: `gnn` the graph trunk's widths (arch-scoped), `aux_soft_policy` the second policy head's rows — REQUIRED, `null` the explicit OFF, armed iff `identity.arch_kind` carries the head."""

    gnn: GnnWidthsConfig | None = None
    aux_soft_policy: AuxSoftPolicyConfig | None = Field(default=...)
