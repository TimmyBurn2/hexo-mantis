"""`ModelConfig` — the net's declared shape as config leaves (contract v35); `select_arch` read the trunk widths off two flat keys no schema carried, so every run built at the dataclass defaults."""
from pydantic import Field

from mantis.config.schema._base import StrictModel


class GnnWidthsConfig(StrictModel):
    """The graph trunk's shape, ONE block (the JK-cat readout is `num_layers × hidden` wide): `ge=1` on both, no off value, graph-scoped by `ARCH_SCOPED_KEYS`, read by `mantis.model.arch.select_arch` alone."""

    hidden: int = Field(ge=1)
    num_layers: int = Field(ge=1)


class ModelConfig(StrictModel):
    """The net's declared shape; `gnn` is the graph trunk's widths (arch-scoped, absent = the key is absent)."""

    gnn: GnnWidthsConfig | None = None
