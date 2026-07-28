"""Non-encoding training/selfplay hyperparameters.

The v6 encoding constants (BOARD_SIZE / NUM_CELLS / BUFFER_CHANNELS /
KEPT_PLANE_INDICES) do NOT live here. The canonical source of truth for every
encoding is the registry at ``crates/mantis-encoding/src/registry.toml``; route
through ``mantis.encoding.lookup(name)`` (Python) or ``mantis_encoding::lookup``
(Rust).

What remains here are non-encoding hyperparameters — values that are not
geometry / plane-layout / action-space attributes of any encoding and
therefore have no place in the registry.
"""

# AlphaZero history length (current + 7 prior timesteps). Self-play /
# training hyperparameter, not an encoding parameter — kept here.
#
# SINGLE SoT for HISTORY_LEN (env.game_state imports this; there is no second
# independent definition). LOAD-BEARING COUPLING: the 18-plane source layout
# places HISTORY_LEN my-stone history planes (source 0..7) then the opponent
# block starting at source plane HISTORY_LEN — i.e. HISTORY_LEN ==
# OPP_STONE_PLANE (== resolvers._OPP_STONE_SRC_PLANE == 8 == the mantis-core
# OPP_STONE_PLANE). A future history-depth change here MUST move the
# opponent-block offset in lockstep; test_b4_history_len_sot pins the equality.
HISTORY_LEN: int = 8

# Depth of the per-worker rolling draw-outcome window the self-play pool keeps
# (`selfplay.instrumentation.PoolInstrumentation._per_worker_draws`, one append per
# completed game). Self-play telemetry geometry, not an encoding parameter — kept here for
# the same reason HISTORY_LEN is: it is a SINGLE SoT read by two layers that must not drift.
#
# LOAD-BEARING COUPLING (WPAX Phase D R80/R83, RE-EXPRESSED at WPMINT Phase DS by R92):
# the config-authored evidence bar `train.draw_rate_abort.N_pool_min` is bounded by this
# window TIMES the worker count. The abort's statistic is the pooled count-weighted rate
# over the UNION of the per-worker windows, so the evidence available to it is
# `sum(len(dq))`, and each deque's `maxlen` is this constant — measured to saturate at
# `DRAW_RATE_WINDOW * n_workers` at 1/2/8/32 workers. A bar above that ceiling is a
# condition no history can satisfy, so the abort audits ARMED and can never fire ("armed in
# the config, absent in effect"). The bound therefore CANNOT be an `le=` on the schema
# field the way `min_samples` (R92-DELETED) carried one: it spans two sections, and it
# lives in `config/schema/core.py::_draw_rate_evidence_bar_is_reachable`. A second
# independent literal on either side would re-open that dead zone silently, which is why
# both sides import THIS name. `tests/selfplay/test_drawrate_pooled_statistic.py` asserts
# the equality against the real deque rather than against the number.
DRAW_RATE_WINDOW: int = 50
