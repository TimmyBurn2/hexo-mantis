"""Policy-scatter LUTs for 12-fold hex symmetry augmentation.

`scatter[dst_flat] = src_flat` — apply via `new_policy = policy[scatter]`.
Matches the Rust SymTables convention: axial (q, r) with q = flat // N - half,
r = flat % N - half; reflect swaps (q, r) → (r, q); rotate applies
(q, r) → (−r, q+r) n_rot times. sym_idx in [0, 12): syms 0-5 are pure
rotations (0–5×60°); syms 6-11 are reflect-then-rotate.
"""
from __future__ import annotations

import numpy as np

from mantis.encoding import lookup as _lookup_encoding

_V6_BOARD_SIZE: int = _lookup_encoding("v6").board_size

_policy_scatters_cache: list[np.ndarray] | None = None

# ── R245: window-preserving subgroup (Python mirror of
# crates/mantis-selfplay/src/replay/sym.rs::WINDOW_PRESERVING_SYMS — that file carries the
# geometric derivation; not re-derived here to avoid a second source of truth for the same
# proof) ──────────────────────────────────────────────────────────────────────────────────
# The D6 elements under which the square axial window is closed: identity, the 180° rotation,
# the axis swap, and their product — a Klein four-group, spelled `sym_idx` 0, 3, 6, 9 under
# this file's `reflect = sym_idx >= 6` / `n_rot = sym_idx % 6` encoding. The other eight
# elements make a cell that leaves the window alias onto the `n_cells - 1` fallback source
# above, injecting label noise rather than augmenting (R245 / ADJ-D27) — but ONLY for a row
# that actually carries content on such a cell. Under R245 ruled option (c) the two Python
# window-clamped DENSE draw sites (`train/batch_assembly.py::_augment_recent_rows`,
# `train/pretrain/dataset.py`'s collate) therefore gate PER ROW via `draw_record_syms` +
# `spread_mask` below: a row that fits the window keeps all 12, a row that does not is
# restricted to this set. Uniform-over-12 at a dense site is never correct.
WINDOW_PRESERVING_SYMS: tuple[int, ...] = (0, 3, 6, 9)


def get_policy_scatters(
    board_size: int = _V6_BOARD_SIZE,
    has_pass: bool = True,
) -> list[np.ndarray]:
    """Return 12 policy-scatter index arrays.

    Each array has length `board_size**2 + (1 if has_pass else 0)` and dtype
    int64. When `has_pass` is True (the pass-slot default), index
    `board_size**2` is the pass move and is invariant under all symmetries.
    When `has_pass` is False (no pass slot), the pass-row scatter entry is
    omitted entirely. Cells whose source maps outside the window fall back to
    `src_flat = board_size**2 - 1` (consistent with the Rust sample_batch
    scatter for out-of-window cells).

    Cached on first call for the canonical default config
    (board_size=_V6_BOARD_SIZE, has_pass=True). Non-canonical callers do not hit
    the cache.
    """
    global _policy_scatters_cache
    is_canonical_default = board_size == _V6_BOARD_SIZE and has_pass
    if _policy_scatters_cache is not None and is_canonical_default:
        return _policy_scatters_cache

    N = board_size
    half = (N - 1) // 2
    n_cells = N * N
    n_actions = n_cells + (1 if has_pass else 0)
    scatters: list[np.ndarray] = []

    for sym_idx in range(12):
        reflect = sym_idx >= 6
        n_rot = sym_idx % 6
        scatter = np.full(n_actions, n_cells - 1, dtype=np.int64)
        if has_pass:
            scatter[n_cells] = n_cells  # pass is invariant
        for src in range(n_cells):
            q = src // N - half
            r = src % N - half
            if reflect:
                q, r = r, q
            for _ in range(n_rot):
                q, r = -r, q + r
            dq = q + half
            dr = r + half
            if 0 <= dq < N and 0 <= dr < N:
                scatter[dq * N + dr] = src
        scatters.append(scatter)

    if is_canonical_default:
        _policy_scatters_cache = scatters
    return scatters


def draw_window_preserving_syms(n: int) -> np.ndarray:
    """Draw `n` D6 element indices uniformly from `WINDOW_PRESERVING_SYMS` (R245).

    The correct draw for every window-clamped DENSE augmentation site whose subject rows
    cannot be certified lossless; see `WINDOW_PRESERVING_SYMS`. A site that CAN certify its
    rows uses `draw_record_syms` and recovers the full group for the rows that fit the window.
    Uses the numpy global RNG (`np.random.randint`), matching the call sites this replaces —
    under a fixed seed the RNG stream consumption differs from the old
    `np.random.randint(0, 12, size=n)` draw (4-wide vs 12-wide draw), same as the Rust-side
    R245 restriction (ADJ-D29).
    """
    subgroup = np.asarray(WINDOW_PRESERVING_SYMS, dtype=np.int64)
    return subgroup[np.random.randint(0, len(subgroup), size=n)]


# ── R245(c): the per-record losslessness gate (Python mirror of
# crates/mantis-selfplay/src/replay/{sym.rs::dropped_cells, mod.rs::slot_is_compact}) ────────
_dropped_cells_cache: dict[int, np.ndarray] = {}


def get_dropped_cells(board_size: int = _V6_BOARD_SIZE) -> np.ndarray:
    """Return **D** — the flat source cells the window-DROPPING D6 elements delete.

    DERIVED, never transcribed: the FORWARD transform this module's LUT builder runs
    (`q = flat // N - half`, `r = flat % N - half`; reflect swaps, then `n_rot` × the 60°
    step) is applied to every cell under every element, and a cell whose image leaves the
    window is dropped. D is the union over all 12 elements — the four window-preserving ones
    contribute nothing, and the other eight in fact contribute the identical set (each of
    their matrices carries one `(±1, ±1)` row, so the surviving condition is `|q + r| <= half`
    for all of them); `tests/train/test_window_preserving_dense_draw.py` pins both facts.

    NOTE it is derived from the TRANSFORM, not from `get_policy_scatters`. The scatter LUT is
    dst→src with an `n_cells - 1` fallback source for destinations nothing maps to, so the
    corner cell (which IS in D) appears in every LUT and a LUT-derived D would silently omit
    it (ADJ-D28). Omitting a cell from D is the UNSOUND direction: a row carrying content only
    there would be called compact and then clipped.

    Cached per board_size (D depends on nothing else).
    """
    cached = _dropped_cells_cache.get(board_size)
    if cached is not None:
        return cached

    N = board_size
    half = (N - 1) // 2
    flat = np.arange(N * N, dtype=np.int64)
    q0 = flat // N - half
    r0 = flat % N - half

    dropped = np.zeros(N * N, dtype=bool)
    for sym_idx in range(12):
        q, r = (r0, q0) if sym_idx >= 6 else (q0, r0)
        for _ in range(sym_idx % 6):
            q, r = -r, q + r
        inside = (q + half >= 0) & (q + half < N) & (r + half >= 0) & (r + half < N)
        dropped |= ~inside

    out = np.flatnonzero(dropped).astype(np.int64)
    _dropped_cells_cache[board_size] = out
    return out


def spread_mask(
    board_size: int,
    *,
    states: np.ndarray | None = None,
    policies: np.ndarray | None = None,
    ownership: np.ndarray | None = None,
    winning_line: np.ndarray | None = None,
) -> np.ndarray:
    """Per-row SPREAD flag (R245(c)): True where a dropping element would clip the row.

    A row is COMPACT (False) iff every channel passed here equals its NEUTRAL on every cell
    of `get_dropped_cells(board_size)`, and SPREAD (True) otherwise. Pass exactly the channels
    the calling site SCATTERS — a channel that is recomputed downstream from an already-checked
    one (chain planes, which `_augment_recent_rows` and the pretrain collate both rebuild from
    the augmented stone planes) is induced by that channel and needs no check of its own.

    NEUTRALS, matching `ReplayBuffer::build`'s column init on the Rust side: states 0,
    policy 0.0, ownership **1** (= empty, NOT 0 — 0 means "owned by P2"), winning_line 0. The
    policy pass slot (index `n_cells`) is positionally invariant under every element, so it is
    never consulted — D holds only cell indices.

    Args:
        board_size: trunk side length (the LUT/geometry authority for D).
        states:     (n, C, H, W) or (n, C * n_cells) — any non-zero on D is content.
        policies:   (n, n_cells [+ pass]) — any non-zero on D is content.
        ownership:  (n, n_cells) — any value != 1 on D is content.
        winning_line: (n, n_cells) — any non-zero on D is content.

    Returns:
        (n,) bool array. Rows are independent; the gate is per record.
    """
    dropped = get_dropped_cells(board_size)
    n_cells = board_size * board_size

    channels = (states, policies, ownership, winning_line)
    first = next((c for c in channels if c is not None), None)
    if first is None:
        raise ValueError("spread_mask: at least one channel must be supplied")
    n = int(np.asarray(first).shape[0])
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        # An empty batch has no row to certify, so the empty mask IS the answer — and it must
        # be returned BEFORE the reshapes below, which numpy cannot resolve at size 0 (it
        # refuses to infer the `-1` dimension there and raises). Returning here keeps this
        # function total over the same domain the flat draws it replaced covered
        # (`np.random.randint(0, 12, size=0)` is empty and every per-row loop is a no-op), so a
        # caller handing over an empty batch gets a no-op rather than a crash. Derived from `n`
        # — NOT a configured special case, and it does not swallow the genuine misuse error
        # above (no channel supplied at all still raises).
        return mask

    if states is not None:
        flat_states = np.asarray(states).reshape(n, -1, n_cells)
        mask |= np.any(flat_states[:, :, dropped] != 0, axis=(1, 2))
    if policies is not None:
        mask |= np.any(np.asarray(policies).reshape(n, -1)[:, dropped] != 0, axis=1)
    if ownership is not None:
        mask |= np.any(np.asarray(ownership).reshape(n, -1)[:, dropped] != 1, axis=1)
    if winning_line is not None:
        mask |= np.any(np.asarray(winning_line).reshape(n, -1)[:, dropped] != 0, axis=1)
    return mask


def draw_record_syms(spread: np.ndarray) -> np.ndarray:
    """Draw one D6 element per row, gated on that row's SPREAD flag (R245(c)).

    A compact row (`spread[i]` False) draws over the full 12-element group — every element
    transforms it exactly. A spread row draws only from `WINDOW_PRESERVING_SYMS`, the elements
    that are lossless unconditionally. Either way NO clipped copy is produced.

    Uses the numpy global RNG, like the flat draws it replaces. Both candidate streams are
    drawn for every row (vectorized `np.where`), so the RNG consumption is row-count-driven and
    independent of the mask — a run's stream does not shift when the data's compactness does.
    """
    spread_bool = np.asarray(spread, dtype=bool)
    n = int(spread_bool.shape[0])
    full = np.random.randint(0, 12, size=n).astype(np.int64)
    restricted = draw_window_preserving_syms(n)
    return np.where(spread_bool, restricted, full).astype(np.int64)
