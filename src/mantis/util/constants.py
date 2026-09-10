"""Non-encoding training/selfplay hyperparameters.

Geometry, plane-layout and action-space values are NOT here: the registry at
``crates/mantis-encoding/src/registry.toml`` is their source of truth, reached through
``mantis.encoding.lookup(name)`` or ``mantis_encoding::lookup``.
"""

# AlphaZero history length (current + 7 prior timesteps), the single definition:
# `env.game_state` imports this, and what it bounds is the `move_history` deque depth.
HISTORY_LEN: int = 8

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
