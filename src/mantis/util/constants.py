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
