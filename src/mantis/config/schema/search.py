"""`SearchConfig` — the run's search regime, as ONE closed identity key.

WHY IT IS ITS OWN SECTION AND NOT A `selfplay` KEY. The search a run performs is not a
self-play knob: `mantis.arena.deploy_head` searches with it too, and the promotion bar's
whole claim (LAW-15, "deploy-matched") is that the two are the same search. A key living
under `selfplay` that the eval head also had to read would make the eval side a reader of
another surface's section, which is exactly how the deploy head came to run a regime the
run never declared.

WHY ONE KEY AND NOT FOUR. `search.kind` replaces `selfplay.gumbel_mcts`,
`selfplay.gumbel_variant`, `selfplay.completed_q_values` and `train.completed_q_values`.
Those four spelled sixteen regimes; two were ever run and none of the other fourteen was
ever measured. Worse, they could DISAGREE — a config could export completed-Q targets from
a PUCT tree, or run a corrected Gumbel search and train against raw visit counts — and the
cross-section validator that kept them in step was a convention with a validator bolted on,
not a single authority. The kind is the authority: the root mechanism, the interior
selector and the exported target's semantics are all read off it.

NO DEFAULT, anywhere (R1/LAW-11). An absent `search.kind` is a mint error, not a silent
PUCT: the regime a run searched under is provenance, and a run whose search regime was
never written down cannot be compared with one whose was.
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
    ``selfplay.c_scale`` = its ``value_scale``) stay where they are and stay REQUIRED with
    no default: their values are a mint decision and the schema will not guess one.
    """

    kind: Literal["puct", "gumbel"]
