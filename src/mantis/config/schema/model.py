"""`ModelConfig` — the net's declared shape as config leaves (contract v35); `select_arch` read the trunk widths off two flat keys no schema carried, so every run built at the dataclass defaults."""
from pydantic import Field

from mantis.config.schema._base import StrictModel


class GnnWidthsConfig(StrictModel):
    """The graph trunk's shape, ONE block (the JK-cat readout is `num_layers × hidden` wide): `ge=1` on both, no off value, graph-scoped by `ARCH_SCOPED_KEYS`, read by `mantis.model.arch.select_arch` alone."""

    hidden: int = Field(ge=1)
    num_layers: int = Field(ge=1)


class AuxSoftPolicyConfig(StrictModel):
    """The auxiliary soft-policy head's loss rows (R366(b), contract v36): the searched target's explicit entries raised to `1/temperature` and renormalised over the explicit mass (`gt=1`: at 1 the head learns the main head's own target and reads armed while dead; below 1 it sharpens, another lever), the tail carried as the hard target's; `weight` (`gt=0`) scales its CE into the step's loss."""

    target_temperature: float = Field(gt=1.0)
    weight: float = Field(gt=0.0)


class ModelConfig(StrictModel):
    """The net's declared shape; `gnn` is the graph trunk's widths (arch-scoped, absent = the key is absent), `aux_soft_policy` the second policy head's rows — REQUIRED, `null` the explicit OFF, present iff `identity.arch_kind` names a kind that carries the head."""

    gnn: GnnWidthsConfig | None = None
    aux_soft_policy: AuxSoftPolicyConfig | None = Field(default=...)
