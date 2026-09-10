"""`SearchConfig` — the run's search regime, as one closed identity key.

It is its own section rather than a `selfplay` key because the deploy head searches with it
too, and the promotion bar's whole claim is that the two searches are the same one.

One key rather than four: the root mechanism, the interior selector and the exported target's
semantics are all read off `search.kind`, so they cannot disagree the way four independent
flags could (completed-Q targets exported from a PUCT tree, say).

There is no default anywhere. An absent `search.kind` is a mint error, not a silent PUCT: a
run whose search regime was never written down cannot be compared with one whose was.
"""
from typing import Literal

from mantis.config.schema._base import StrictModel


class SearchConfig(StrictModel):
    """The search regime.

    ``kind``:

    * ``puct`` — PUCT descent at every node, Dirichlet root noise, and the
      temperature-annealed VISIT distribution as the training target.
    * ``gumbel`` — Gumbel-Top-k root sampling with Sequential Halving over the full legal
      prior, completed-Q interior selection, no Dirichlet (the Gumbel draw IS the root
      exploration), and the completed-Q IMPROVED POLICY as the training target. Follows
      `google-deepmind/mctx`'s ``gumbel_muzero_policy``; pinned against Mctx's own outputs
      by ``crates/mantis-search/tests/mctx_parity.rs``.

    The Q-scale knobs the kind spends (``selfplay.c_visit`` = Mctx's ``maxvisit_init``,
    ``selfplay.c_scale`` = its ``value_scale``) stay required with no default: their values
    are a mint decision and the schema will not guess one.
    """

    kind: Literal["puct", "gumbel"]
