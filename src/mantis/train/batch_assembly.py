"""Replay-buffer → training-batch assembly (WP10 §a.4 PORT; old training/batch_assembly.py).

>300 justify: the mixed-batch assembly path is ONE concern — draw corpus + self-play (+ optional
recent-window) rows, augment the Python-side recent rows, and pack them into pre-allocated buffers
(steady-state) or concatenate (warm-up). Kept together so the batch-layout contract
(`[corpus | bot | recent | uniform_self]`) is greppable. Behaviour-exact relocation, routed
through the new-repo seams: `mantis.data.augment.get_policy_scatters`, `mantis.env.game_state.
_compute_chain_planes`, `mantis.encoding` slot/pin helpers, `mantis.data.corpus_io` sha/sidecar,
and `mantis._engine` (`ReplayBuffer`, `apply_symmetries_batch`) — the self-play buffer sampling is
the injected engine seam. train → data / env / encoding are all DAG-legal.

RESERVED, NOT DEAD (R289(q)) — AND THE CORPUS LOADER IS NO LONGER PART OF THE RESERVE
(R326(d)). At run5 this module's mixing arms receive nothing: `run.py` passes
`pretrained_buffer=None`, `recent_buffer=None` and `bufs=None` into `StepCoordinator`, so the
corpus/bot/recent slots are empty and only the self-play rows flow. The config keys and resolver
stamps that feed those slots ARE live, which is what makes the gap look like dead code to a
census. R289(q) holds the mixed-batch path RESERVED on the ground that it is one arm of an
operator decision not yet taken (queue row RQ-19), **and the ASSEMBLY arms below still stand on
that ground.** What no longer does is the loader that fed them: the operator VALUED the
BOOTSTRAP POSTURE at (A), BC-pretrain with no anchor, and CLOSED the row, so posture (B) —
corpus-mix — is excluded by definition and `load_pretrained_buffer`'s reason to exist expired
with it. R326(d) deleted it; see the grave marker below. **The supersession is PARTIAL and
deliberate: the loader is gone, the mixing arms are not.**

UNREACHABLE ON A GRAPH RUN, IN THREE INDEPENDENT PLACES (R325(b), verified at 3bde2d1) — two of
which survive the deletion and one of which the deletion CONSUMED. The un-wiring above is the
RUN5 reading and it understates the fact. (1) `StepCoordinator`'s mixed arm refuses a non-`grid`
representation at the route — STANDS. (2) the corpus loader was dense-only, an NPZ read against
a graph run's `.hexg` ring — GONE WITH THE LOADER, and R326(d) is why the R325(c) contract
refusal that had just been added to it is not in this file any more. (3) Nothing called it —
which is what made the deletion a deletion rather than a removal of a feature. So
`train.bot_batch_share: 0.0` and the `train.mixing.*` values in every shipped config remain
STRUCTURAL FACT, not a tuning preference: on a graph run there is nothing for a non-zero value
to reach, and now there is not even a loader to reach it with.

Scope note (DESIGN deviation, documented): the bot-corpus ATOMIC-SWAP machinery
(`swap_bot_corpus_atomic` / `BotCorpusSwapError` / `load_bot_corpus_buffer`) is bot-refresh-
adjacent (the KILLED `bot_refresh` subprocess family swaps the bot NPZ) — it is NOT ported. The
`bot_batch_share` MIXING slot survives (a bot buffer, when injected, still contributes rows here);
only the runtime REGEN/SWAP hook (KILL) is dropped.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mantis.encoding import lookup as _lookup_encoding
from mantis.encoding import opp_stone_slot

_LOG = logging.getLogger(__name__)

_V6 = _lookup_encoding("v6")
_V6_BOARD_SIZE: int = _V6.board_size
_V6_BUFFER_CHANNELS: int = _V6.n_planes


# ── Mixed-batch result + pre-allocated buffers ───────────────────────────────────────────
@dataclass(frozen=True)
class BatchAssemblyResult:
    """Result of :func:`assemble_mixed_batch` — eight arrays (views into `BatchBuffers` in
    steady state, freshly allocated in warm-up) + `n_recent_actual` so callers can slice
    `[corpus | (bot) | recent | uniform_self]`."""

    states: np.ndarray
    chain_planes: np.ndarray
    policies: np.ndarray
    outcomes: np.ndarray
    ownership: np.ndarray
    winning_line: np.ndarray
    is_full_search: np.ndarray
    n_recent_actual: int
    position_indices: np.ndarray | None = None
    value_target_valid: np.ndarray | None = None


@dataclass
class BatchBuffers:
    """Pre-allocated arrays reused each step (no per-step malloc/free). `warmup_active` flips
    False the first time all sources return the full requested row count."""

    states: np.ndarray
    chain_planes: np.ndarray
    policies: np.ndarray
    outcomes: np.ndarray
    ownership: np.ndarray
    winning_line: np.ndarray
    is_full_search: np.ndarray
    value_target_valid: np.ndarray
    warmup_active: bool = field(default=True)


def allocate_batch_buffers(
    batch_size: int,
    n_actions: int,
    trunk_size: int = _V6_BOARD_SIZE,
    aux_stride: int | None = None,
    n_planes: int = _V6_BUFFER_CHANNELS,
) -> BatchBuffers:
    """Allocate the shared batch arrays once at startup (encoding-derived spatial shapes)."""
    if aux_stride is None:
        aux_stride = trunk_size * trunk_size  # noqa: F841 — reserved hook
    return BatchBuffers(
        states=np.empty((batch_size, n_planes, trunk_size, trunk_size), dtype=np.float16),
        chain_planes=np.empty((batch_size, 6, trunk_size, trunk_size), dtype=np.float16),
        policies=np.empty((batch_size, n_actions), dtype=np.float32),
        outcomes=np.empty(batch_size, dtype=np.float32),
        ownership=np.empty((batch_size, trunk_size, trunk_size), dtype=np.uint8),
        winning_line=np.empty((batch_size, trunk_size, trunk_size), dtype=np.uint8),
        is_full_search=np.ones(batch_size, dtype=np.uint8),
        value_target_valid=np.ones(batch_size, dtype=np.uint8),
    )


# GRAVE (R326(d), 2026-08-31): `load_pretrained_buffer` stood here — the dense-only corpus-NPZ
# feed for posture (B), corpus-mix. The operator VALUED the bootstrap posture (A) and CLOSED
# the row, so (B) is excluded by definition and R289(q)'s reserve, whose ground was that the
# DECISION was untaken, no longer reaches it. Zero call sites when it went; the absence is
# pinned by an import census in `tests/train/test_bc_graph_reroute.py`.

# ── recent-row augmentation (Python-side; RecentBuffer skips the Rust sample kernel) ──────
def _augment_recent_rows(
    s_r: np.ndarray,
    c_r: np.ndarray,
    p_r: np.ndarray,
    own_r_flat: np.ndarray,
    wl_r_flat: np.ndarray,
    augment: bool,
    opp_slot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply hex augmentation to RecentBuffer rows in Python (they skip the Rust sample
    kernel), drawn PER ROW over the group that is lossless for that row (R245(c) — this is a
    window-clamped DENSE site; see `mantis.data.augment.spread_mask`): Rust
    `apply_symmetries_batch` for stone planes (v6 fast path) or a pure-numpy scatter (v6w25),
    chain planes recomputed from augmented stones, numpy scatter for targets."""
    if not augment:
        return s_r, c_r, p_r, own_r_flat, wl_r_flat

    import mantis._engine as _engine
    from mantis.data.augment import draw_record_syms, get_policy_scatters, spread_mask
    from mantis.env.game_state import _compute_chain_planes

    n = len(s_r)
    board_size = int(s_r.shape[-1])
    n_cells = board_size * board_size
    has_pass = p_r.shape[1] == n_cells + 1
    scatters = get_policy_scatters(board_size, has_pass=has_pass)
    # R245(c): certify each row BEFORE augmenting, over exactly the channels scattered below.
    # `c_r` is deliberately absent — the chain planes are RECOMPUTED post-augment from the
    # augmented stone planes, so their content on D is induced by `s_r` and already covered.
    sym_indices = draw_record_syms(
        spread_mask(
            board_size, states=s_r, policies=p_r, ownership=own_r_flat, winning_line=wl_r_flat
        )
    )

    states_f32 = s_r.astype(np.float32)
    if board_size == _V6_BOARD_SIZE:
        states_f32 = _engine.apply_symmetries_batch(states_f32, sym_indices.tolist())
        s_r = states_f32.astype(np.float16)
    else:
        C = states_f32.shape[1]
        spatial = n_cells
        states_flat = states_f32.reshape(n, C, spatial)
        augmented = np.empty_like(states_flat)
        for sym in range(12):
            mask_idx = np.where(sym_indices == sym)[0]
            if mask_idx.size == 0:
                continue
            sc = scatters[sym]
            state_scatter = sc[:spatial] if has_pass else sc
            augmented[mask_idx] = states_flat[mask_idx][:, :, state_scatter]
        states_f32 = augmented.reshape(n, C, board_size, board_size)
        s_r = states_f32.astype(np.float16)

    c_r_aug = np.empty_like(c_r)
    for i in range(n):
        c_r_aug[i] = (
            _compute_chain_planes(states_f32[i, 0], states_f32[i, opp_slot]).astype(np.float32) / 6.0
        ).astype(np.float16)

    scattered_p = np.empty_like(p_r)
    scattered_own = np.empty_like(own_r_flat)
    scattered_wl = np.empty_like(wl_r_flat)
    for i in range(n):
        lut = scatters[int(sym_indices[i])]
        scattered_p[i] = p_r[i][lut]
        scattered_own[i] = own_r_flat[i][lut[:n_cells]]
        scattered_wl[i] = wl_r_flat[i][lut[:n_cells]]

    return s_r, c_r_aug, scattered_p, scattered_own, scattered_wl


# ── mixed-batch assembly ───────────────────────────────────────────────────────────────────
def assemble_mixed_batch(
    pretrained_buffer: Any,
    buffer: Any,
    recent_buffer: Any | None,
    n_pre: int,
    n_self: int,
    batch_size: int,
    # None = the coordinator was built without a config batch size (its default); it
    # compares unequal below and keeps the warm-up concat path, same as any mismatch.
    batch_size_cfg: int | None,
    recency_weight: float,
    bufs: BatchBuffers,
    train_step: int,
    augment: bool = True,
    *,
    bot_buffer: Any | None = None,
    n_bot: int = 0,
) -> BatchAssemblyResult:
    """Assemble one mixed batch from pretrain + (bot) + self-play (+ optional recent) buffers.

    Warm-up (partial buffers) → `np.concatenate` (allocates); steady-state (all sources full)
    → in-place `np.copyto` into `bufs` + clears `warmup_active`. Batch order:
    `[corpus | bot | recent | uniform_self]` (bot active) else `[corpus | recent | uniform_self]`.
    Corpus AND bot rows always have `is_full_search=1`. Behaviour-exact."""
    s_pre, c_pre, p_pre, o_pre, own_pre, wl_pre, ifs_pre, pos_pre, vv_pre = \
        pretrained_buffer.sample_batch_with_pos(n_pre, augment)

    use_bot = bot_buffer is not None and n_bot > 0 and bot_buffer.size > 0
    if use_bot:
        assert bot_buffer is not None  # use_bot's first conjunct, restated for the checker
        s_b, c_b, p_b, o_b, own_b, wl_b, _ifs_b, pos_b, _vv_b = \
            bot_buffer.sample_batch_with_pos(n_bot, augment)
        ifs_b = np.ones(len(s_b), dtype=np.uint8)
        vv_b = np.ones(len(s_b), dtype=np.uint8)
    else:
        n_bot = 0
        s_b = c_b = p_b = o_b = own_b = wl_b = ifs_b = pos_b = vv_b = None

    if batch_size != batch_size_cfg:
        if train_step > 100:
            _LOG.warning("mixed_batch_size_mismatch batch_size=%s expected=%s", batch_size, batch_size_cfg)
        s_self, c_self, p_self, o_self, own_self, wl_self, ifs_self, pos_self, vv_self = \
            _sample_selfplay(buffer, recent_buffer, n_self, recency_weight, augment)
        parts = ([(s_pre, c_pre, p_pre, o_pre, own_pre, wl_pre, ifs_pre, pos_pre, vv_pre)]
                 + ([(s_b, c_b, p_b, o_b, own_b, wl_b, ifs_b, pos_b, vv_b)] if use_bot else [])
                 + [(s_self, c_self, p_self, o_self, own_self, wl_self, ifs_self, pos_self, vv_self)])
        return _concat_result(parts, n_recent_actual=0)

    use_recent = (recent_buffer is not None and recent_buffer.size > 0
                  and recency_weight > 0.0 and n_self > 1)
    bot_piece = (s_b, c_b, p_b, o_b, own_b, wl_b, ifs_b, pos_b, vv_b) if use_bot else None

    n_recent_actual = 0
    if use_recent:
        assert recent_buffer is not None  # use_recent's first conjunct, restated for the checker
        n_recent_req = max(1, int(round(n_self * recency_weight)))
        n_uniform = n_self - n_recent_req
        s_r, c_r, p_r, o_r, own_r_flat, wl_r_flat, ifs_r, vv_r = recent_buffer.sample(n_recent_req)
        s_r, c_r, p_r, own_r_flat, wl_r_flat = _augment_recent_rows(
            s_r, c_r, p_r, own_r_flat, wl_r_flat, augment, opp_stone_slot(buffer.encoding))
        _bs = int(s_r.shape[-1])
        own_r = own_r_flat.reshape(-1, _bs, _bs)
        wl_r = wl_r_flat.reshape(-1, _bs, _bs)
        pos_r = np.zeros(len(s_r), dtype=np.uint16)
        s_u, c_u, p_u, o_u, own_u, wl_u, ifs_u, pos_u, vv_u = \
            buffer.sample_batch_with_pos(max(1, n_uniform), augment)
        n_recent_actual = len(s_r)
        pieces = [(s_pre, c_pre, p_pre, o_pre, own_pre, wl_pre, ifs_pre, pos_pre, vv_pre)]
        if bot_piece is not None:
            pieces.append(bot_piece)
        pieces.extend([(s_r, c_r, p_r, o_r, own_r, wl_r, ifs_r, pos_r, vv_r),
                       (s_u, c_u, p_u, o_u, own_u, wl_u, ifs_u, pos_u, vv_u)])
        n_avail = n_pre + (n_bot if use_bot else 0) + len(s_r) + len(s_u)
    else:
        s_u, c_u, p_u, o_u, own_u, wl_u, ifs_u, pos_u, vv_u = \
            buffer.sample_batch_with_pos(max(1, n_self), augment)
        pieces = [(s_pre, c_pre, p_pre, o_pre, own_pre, wl_pre, ifs_pre, pos_pre, vv_pre)]
        if bot_piece is not None:
            pieces.append(bot_piece)
        pieces.append((s_u, c_u, p_u, o_u, own_u, wl_u, ifs_u, pos_u, vv_u))
        n_avail = n_pre + (n_bot if use_bot else 0) + len(s_u)

    if n_avail < batch_size:
        return _concat_result(pieces, n_recent_actual=n_recent_actual)

    if bufs.warmup_active:
        _LOG.info("buffer_warmup_ended step=%s n_available=%s batch_size=%s",
                  train_step, n_avail, batch_size)
        bufs.warmup_active = False

    out_pos = np.concatenate([p[7] for p in pieces], axis=0)
    offset = 0
    for s, c, p, o, own, wl, ifs, _pos, vv in pieces:
        n = len(s)
        np.copyto(bufs.states[offset:offset + n], s)
        np.copyto(bufs.chain_planes[offset:offset + n], c)
        np.copyto(bufs.policies[offset:offset + n], p)
        np.copyto(bufs.outcomes[offset:offset + n], o)
        np.copyto(bufs.ownership[offset:offset + n], own)
        np.copyto(bufs.winning_line[offset:offset + n], wl)
        np.copyto(bufs.is_full_search[offset:offset + n], ifs)
        np.copyto(bufs.value_target_valid[offset:offset + n], vv)
        offset += n

    return BatchAssemblyResult(
        states=bufs.states, chain_planes=bufs.chain_planes, policies=bufs.policies,
        outcomes=bufs.outcomes, ownership=bufs.ownership, winning_line=bufs.winning_line,
        is_full_search=bufs.is_full_search, n_recent_actual=n_recent_actual,
        position_indices=out_pos, value_target_valid=bufs.value_target_valid,
    )


def _concat_result(pieces: list[tuple[Any, ...]], *, n_recent_actual: int) -> BatchAssemblyResult:
    return BatchAssemblyResult(
        states=np.concatenate([p[0] for p in pieces], axis=0),
        chain_planes=np.concatenate([p[1] for p in pieces], axis=0),
        policies=np.concatenate([p[2] for p in pieces], axis=0),
        outcomes=np.concatenate([p[3] for p in pieces], axis=0),
        ownership=np.concatenate([p[4] for p in pieces], axis=0),
        winning_line=np.concatenate([p[5] for p in pieces], axis=0),
        is_full_search=np.concatenate([p[6] for p in pieces], axis=0),
        n_recent_actual=n_recent_actual,
        position_indices=np.concatenate([p[7] for p in pieces], axis=0),
        value_target_valid=np.concatenate([p[8] for p in pieces], axis=0),
    )


def _sample_selfplay(
    buffer: Any,
    recent_buffer: Any | None,
    n_self: int,
    recency_weight: float,
    augment: bool = True,
) -> tuple[np.ndarray, ...]:
    """Sample self-play rows, blending recent + uniform when recency_weight > 0 (9-tuple:
    states, chain, policies, outcomes, ownership, winning_line, is_full_search,
    position_indices, value_target_valid)."""
    if (recent_buffer is not None and recent_buffer.size > 0
            and recency_weight > 0.0 and n_self > 1):
        n_r = max(1, int(round(n_self * recency_weight)))
        n_u = n_self - n_r
        s_r, c_r, p_r, o_r, own_r_flat, wl_r_flat, ifs_r, vv_r = recent_buffer.sample(n_r)
        s_r, c_r, p_r, own_r_flat, wl_r_flat = _augment_recent_rows(
            s_r, c_r, p_r, own_r_flat, wl_r_flat, augment, opp_stone_slot(buffer.encoding))
        _bs = int(s_r.shape[-1])
        own_r = own_r_flat.reshape(-1, _bs, _bs)
        wl_r = wl_r_flat.reshape(-1, _bs, _bs)
        pos_r = np.zeros(len(s_r), dtype=np.uint16)
        s_u, c_u, p_u, o_u, own_u, wl_u, ifs_u, pos_u, vv_u = \
            buffer.sample_batch_with_pos(max(1, n_u), augment)
        return (
            np.concatenate([s_r, s_u], axis=0), np.concatenate([c_r, c_u], axis=0),
            np.concatenate([p_r, p_u], axis=0), np.concatenate([o_r, o_u], axis=0),
            np.concatenate([own_r, own_u], axis=0), np.concatenate([wl_r, wl_u], axis=0),
            np.concatenate([ifs_r, ifs_u], axis=0), np.concatenate([pos_r, pos_u], axis=0),
            np.concatenate([vv_r, vv_u], axis=0),
        )
    return buffer.sample_batch_with_pos(max(1, n_self), augment)
