"""`SearchConfig` — a search regime: its kind and its tactics block — and `DeployConfig`, the deploy head's.

Two homes: `selfplay.search.kind` is what the workers run and what the exported targets
MEAN; `deploy.search.kind` is what the bar plays, matched to what will be deployed.
One key per regime rather than four, so the root, the interior selector and the target cannot
disagree; no default anywhere — an absent `kind` is a mint error, not a silent PUCT.
"""
from typing import Literal

from pydantic import Field

from mantis.config.schema._base import StrictModel

#: The bridge parses a node budget as an `i64`; a larger minted value would validate and then fail to arm.
_MAX_NODES = 2**63 - 1


class TacticsAuditConfig(StrictModel):
    """The defence audit's budgets: a call's `turns` and `nodes`, `k` alternatives, `m` second stones, one move's total."""

    turns: int = Field(ge=1, le=40)
    nodes: int = Field(ge=1, le=_MAX_NODES)
    k: int = Field(ge=1, le=1024)
    m: int = Field(ge=1, le=1024)
    total_nodes: int = Field(ge=1, le=_MAX_NODES)


class TacticsConfig(StrictModel):
    """The tactics module (v39): the strict turn solver's leaf and root budgets and the audit, `null` the explicit off."""

    kind: Literal["strict_turn"]
    leaf_turns: int = Field(ge=1, le=40)
    leaf_nodes: int = Field(ge=0, le=_MAX_NODES)
    root_turns: int = Field(ge=1, le=40)
    root_nodes: int = Field(ge=0, le=_MAX_NODES)
    audit: TacticsAuditConfig | None = Field(default=...)


class SearchConfig(StrictModel):
    """A search regime.

    ``kind``:

    * ``puct`` — PUCT descent at every node, Dirichlet root noise, and the
      temperature-annealed VISIT distribution as the training target.
    * ``gumbel`` — Gumbel-Top-k root sampling with Sequential Halving over the full legal
      prior, completed-Q interior selection, no Dirichlet (the Gumbel draw IS the root
      exploration), and the completed-Q IMPROVED POLICY as the training target. Follows
      `google-deepmind/mctx`'s ``gumbel_muzero_policy``; pinned against Mctx's own outputs
      by ``crates/mantis-search/tests/mctx_parity.rs``.

    The σ the kind spends (``selfplay.{c_visit, c_scale, q_rescale}``) is ONE key set shared
    by both homes; the ``kind`` and the ``tactics`` block may differ between them.
    """

    kind: Literal["puct", "gumbel"]
    tactics: TacticsConfig | None = Field(default=...)


class DeployConfig(StrictModel):
    """The deploy head's regime — the bar's and every external cell's; may differ from self-play's."""

    search: SearchConfig
