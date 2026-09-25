"""`SearchConfig` — a search regime as one closed key — and `DeployConfig`, the deploy head's.

Two homes: `selfplay.search.kind` is what the workers run and what the exported targets
MEAN; `deploy.search.kind` is what the bar plays, matched to what will be deployed.
One key per regime rather than four, so the root, the interior selector and the target cannot
disagree; no default anywhere — an absent `kind` is a mint error, not a silent PUCT.
"""
from typing import Literal

from mantis.config.schema._base import StrictModel


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
    by both homes; only the ``kind`` may differ between them.
    """

    kind: Literal["puct", "gumbel"]


class DeployConfig(StrictModel):
    """The deploy head's regime — the bar's and every external cell's; may differ from self-play's."""

    search: SearchConfig
