"""Replay-buffer → training-batch assembly (WP10 §a.4 PORT; old training/batch_assembly.py).

>300 justify: the mixed-batch assembly path is ONE concern — draw corpus + self-play (+ optional
recent-window) rows, augment the Python-side recent rows, and pack them into pre-allocated buffers
(steady-state) or concatenate (warm-up). Kept together so the batch-layout contract
(`[corpus | bot | recent | uniform_self]`) is greppable. Behaviour-exact relocation, routed
through the new-repo seams: `mantis.data.augment.get_policy_scatters`, `mantis.env.game_state.
_compute_chain_planes`, `mantis.encoding` slot/pin helpers, `mantis.data.corpus_io` sha/sidecar,
and `mantis._engine` (`ReplayBuffer`, `apply_symmetries_batch`) — the self-play buffer sampling is
the injected engine seam. train → data / env / encoding are all DAG-legal.

Scope note (DESIGN deviation, documented): the bot-corpus ATOMIC-SWAP machinery
(`swap_bot_corpus_atomic` / `BotCorpusSwapError` / `load_bot_corpus_buffer`) is bot-refresh-
adjacent (the KILLED `bot_refresh` subprocess family swaps the bot NPZ) — it is NOT ported. The
`bot_batch_share` MIXING slot survives (a bot buffer, when injected, still contributes rows here);
only the runtime REGEN/SWAP hook (KILL) is dropped.
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mantis.data.corpus_io import compute_npz_sha256, validate_corpus_sidecar
from mantis.encoding import (
    assert_not_heldout_sha,
    cur_stone_slot,
    heldout_size_bytes,
    opp_stone_slot,
    resolve_corpus_sha_pin,
)
from mantis.encoding import (
    lookup as _lookup_encoding,
)
from mantis.encoding import (
    normalize_encoding_name as _normalize_encoding_name,
)

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


# ── Corpus loading ───────────────────────────────────────────────────────────────────────
def load_pretrained_buffer(
    mixing_cfg: dict[str, Any],
    config: dict[str, Any],
    emit_fn: Callable[[dict[str, Any]], None],
    buffer_size: int,
    buffer_capacity: int,
) -> Any | None:
    """Load a corpus NPZ into a Rust `ReplayBuffer` with neutral aux padding (ownership=1,
    winning_line=0 — the `n_pretrain` row-slice masks them from aux losses). Behaviour-exact:
    launch-pin sha gate, held-out contamination gate, plane-count check, §102.a chain-plane
    recompute. Returns None when no corpus path is configured (unpinned)."""
    from mantis._engine import ReplayBuffer  # engine only available post-build

    pretrained_path = mixing_cfg.get("pretrained_buffer_path")
    _spec = _lookup_encoding(_normalize_encoding_name(config.get("encoding")))
    _pin = resolve_corpus_sha_pin(_spec)
    _auto_resolved = bool(mixing_cfg.get("_pretrained_buffer_path_auto_resolved"))

    if _auto_resolved and _pin is None:
        raise ValueError(
            f"<auto> corpus for encoding {_spec.name!r} requires a sha pin; add a pin or use "
            "an explicit path.")

    if not pretrained_path or pretrained_path == "<auto>" or not Path(pretrained_path).exists():
        if _pin is not None:
            raise ValueError(
                f"pinned corpus for encoding {_spec.name!r} is missing or unresolved: "
                f"mixing.pretrained_buffer_path={pretrained_path!r}. This corpus is "
                "launch-critical — sync the byte-identical corpus; do NOT proceed corpus-less.")
        if not pretrained_path:
            return None
        _LOG.warning("corpus_npz_not_found path=%s — self-play-only", pretrained_path)
        return None

    # Held-out contamination gate (unconditional; cheap — a metadata stat unless the size matches).
    if os.path.getsize(pretrained_path) in heldout_size_bytes():
        _heldout_check_sha = compute_npz_sha256(pretrained_path)
        assert_not_heldout_sha(_heldout_check_sha, path=pretrained_path)
    else:
        _heldout_check_sha = None

    if _pin is not None:
        _actual_sha = _heldout_check_sha if _heldout_check_sha is not None else compute_npz_sha256(pretrained_path)
        if _actual_sha != _pin:
            raise ValueError(
                f"corpus sha mismatch: {pretrained_path} is {_actual_sha[:12]}…, expected "
                f"{_pin[:12]}… for encoding {_spec.name!r}. Both hosts must read the byte-identical "
                "launch corpus; do NOT re-export.")
        validate_corpus_sidecar(pretrained_path, expected_encoding=_spec.name, actual_sha=_actual_sha)

    t0 = time.time()
    data = np.load(pretrained_path, mmap_mode="r")
    board_size = config.get("board_size", _V6_BOARD_SIZE)
    pre_states = data["states"]
    pre_policies = data["policies"]
    pre_outcomes = data["outcomes"]
    T = len(pre_outcomes)

    _expected_planes = _spec.n_planes
    if pre_states.shape[1] != _expected_planes:
        raise ValueError(
            f"corpus '{pretrained_path}': states.shape[1]={pre_states.shape[1]}, expected "
            f"{_expected_planes} (encoding {_normalize_encoding_name(config.get('encoding'))!r}).")

    # §102.a: recompute chain planes from stone planes (slots derived from the registry).
    from mantis.env.game_state import _compute_chain_planes
    _cur_slot = cur_stone_slot(_spec)
    _opp_slot = opp_stone_slot(_spec)
    pre_chain = np.empty((T, 6, board_size, board_size), dtype=np.float16)
    if T > 0:
        cur_all = np.asarray(pre_states[:, _cur_slot], dtype=np.float32)
        opp_all = np.asarray(pre_states[:, _opp_slot], dtype=np.float32)
        for k in range(T):
            planes_f32 = _compute_chain_planes(cur_all[k], opp_all[k]).astype(np.float32) / 6.0
            pre_chain[k] = planes_f32.astype(np.float16)

    max_pre = int(mixing_cfg.get("pretrain_max_samples", 0))
    if max_pre and T > max_pre:
        _rng = np.random.default_rng(int(config.get("seed", 42)))
        idx = np.sort(_rng.choice(T, size=max_pre, replace=False))
        pre_states, pre_chain = pre_states[idx], pre_chain[idx]
        pre_policies, pre_outcomes = pre_policies[idx], pre_outcomes[idx]
        T = max_pre

    _enc = _normalize_encoding_name(config.get("encoding"))
    pretrained_buffer = ReplayBuffer(capacity=T, encoding=_enc)
    n_cells = board_size * board_size
    pre_own = np.ones((T, n_cells), dtype=np.uint8)
    pre_wl = np.zeros((T, n_cells), dtype=np.uint8)
    pretrained_buffer.push_game(pre_states, pre_chain, pre_policies, pre_outcomes, pre_own, pre_wl)
    del pre_states, pre_chain, pre_policies, pre_outcomes, pre_own, pre_wl, data
    _LOG.info("corpus_loaded positions=%s seconds=%.1f", T, time.time() - t0)
    emit_fn({"event": "system_stats", "buffer_size": buffer_size, "buffer_capacity": buffer_capacity})
    return pretrained_buffer


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
    """Apply 12-fold hex augmentation to RecentBuffer rows in Python (they skip the Rust
    sample kernel): Rust `apply_symmetries_batch` for stone planes (v6 fast path) or a pure-numpy
    scatter (v6w25), chain planes recomputed from augmented stones, numpy scatter for targets."""
    if not augment:
        return s_r, c_r, p_r, own_r_flat, wl_r_flat

    import mantis._engine as _engine
    from mantis.data.augment import get_policy_scatters
    from mantis.env.game_state import _compute_chain_planes

    n = len(s_r)
    board_size = int(s_r.shape[-1])
    n_cells = board_size * board_size
    has_pass = p_r.shape[1] == n_cells + 1
    scatters = get_policy_scatters(board_size, has_pass=has_pass)
    sym_indices = np.random.randint(0, 12, size=n)

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
