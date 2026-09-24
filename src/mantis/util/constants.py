"""Non-encoding training/selfplay hyperparameters.

Geometry, plane-layout and action-space values are NOT here: the registry at
``crates/mantis-encoding/src/registry.toml`` is their source of truth, reached through
``mantis.encoding.lookup(name)`` or ``mantis_encoding::lookup``.
"""

# Depth of the per-worker rolling draw-outcome window the self-play pool keeps, one append
# per completed game.
#
# LOAD-BEARING COUPLING: the evidence bar `train.draw_rate_abort.N_pool_min` is bounded by
# this window TIMES the worker count — the abort's evidence is the union of the per-worker
# deques, measured to saturate at `DRAW_RATE_WINDOW * n_workers` at 1/2/8/32 workers, and a
# bar above that ceiling audits ARMED but can never fire. The bound spans two sections, so it
# cannot be an `le=` on the schema field and lives in
# `config/schema/core.py::_draw_rate_evidence_bar_within_configured_capacity`; both sides
# import THIS name so no second literal can re-open the dead zone silently.
DRAW_RATE_WINDOW: int = 50

# Depth of the pool-wide ring of per-game ply-cap flags (R352(c)'s halt evidence); the same
# coupling as `DRAW_RATE_WINDOW`: `train.ply_cap_abort.window_games` is bounded by it in core.py.
PLY_CAP_RING_GAMES: int = 4096

# A sparse Gumbel row at alpha = 1.0 (within one f32 ULP): ONE authority for the self-play counter
# and the trainer's exclusion (R350(e)) — `selfplay` and `train` may not import each other.
ALPHA_FULL_THRESHOLD: float = 1.0 - 1e-6


def is_alpha_full(alpha: float) -> bool:
    """The ONE predicate behind the counter and the exclusion, in float64 on both sides."""
    return float(alpha) >= ALPHA_FULL_THRESHOLD
