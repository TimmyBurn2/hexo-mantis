"""R245 ruled option (c) — the per-record losslessness gate on the Python DENSE draw sites.

`crates/mantis-selfplay/src/replay/sym.rs::WINDOW_PRESERVING_SYMS` names the order-4 subgroup
`{0, 3, 6, 9}` of D6 under which the square axial window is CLOSED. R245 option (c) rules that
the dense arm draws PER RECORD: a SPREAD record (one a dropping element would clip) gets only
that subgroup, a COMPACT (window-fitting) record gets all 12, and no clipped copy is ever
trained. ADJ-D27 found the two Python window-clamped DENSE draw sites, both originally
`np.random.randint(0, 12, ...)`:

  - `train/batch_assembly.py::_augment_recent_rows` (RecentBuffer rows; `train.augment`-gated,
    false in every shipped config today — inert).
  - `train/pretrain/dataset.py`'s `make_augmented_collate` closure (the bootstrap-pretrain
    path; LIVE via `pretrain/cli.py`'s hardcoded `augment=True`, a separate un-ruled ask).

Both now draw via `mantis.data.augment.draw_record_syms`, gated on
`mantis.data.augment.spread_mask`. This file pins the dropped-cell set D (derived, not
transcribed), the neutral-aware spread mask, the gated draw's support on both arms, the two
sites' BEHAVIOUR (mass conservation under augmentation), and source-presence at both call
sites — mirroring `crates/mantis-selfplay/src/replay/sym.rs`'s own
`dense_sites_restricted_and_graph_site_keeps_the_full_group` pin style: reverting either site
to `np.random.randint(0, 12, ...)` REDs the matching source-presence test below.

R8: over the 300-line soft cap by design. The mechanism is D → the spread mask read against D
→ the draw gated on that mask → the two sites that consume the draw; each layer is only
meaningful against the one below it, and the fixtures (a compact row, a spread row, an
ownership-only spread row) are the shared vocabulary every layer is checked in. Splitting the
helper pins from the site behaviour would let either half stay green while the pipeline is
broken end to end.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import mantis.train.batch_assembly as batch_assembly
import mantis.train.pretrain.dataset as pretrain_dataset
from mantis.data.augment import (
    WINDOW_PRESERVING_SYMS,
    draw_record_syms,
    draw_window_preserving_syms,
    get_dropped_cells,
    get_policy_scatters,
    spread_mask,
)

_BATCH_ASSEMBLY_SRC = Path(batch_assembly.__file__).read_text(encoding="utf-8")
_PRETRAIN_DATASET_SRC = Path(pretrain_dataset.__file__).read_text(encoding="utf-8")

_BOARD = 19
_N_CELLS = _BOARD * _BOARD
_N_PLANES = 8


# ── the constant + shared draw helper ──────────────────────────────────────────────────────
def test_window_preserving_syms_is_the_order_four_subgroup() -> None:
    """Matches the Rust-side `WINDOW_PRESERVING_SYMS` spelling exactly."""
    assert tuple(WINDOW_PRESERVING_SYMS) == (0, 3, 6, 9)


def test_window_preserving_syms_derived_from_scatter_bijectivity() -> None:
    """`WINDOW_PRESERVING_SYMS` is COMPUTED from the scatter geometry — a sym index preserves
    the window iff its cell-block scatter is a bijection of `range(n_cells)` — and compared to
    the constant, never merely a transcribed literal (mirrors
    `crates/mantis-selfplay/src/replay/sym.rs::window_preserving_syms_are_derived_not_asserted`).
    Every element outside the set must genuinely collide (fall back to the shared `n_cells - 1`
    source for more than one destination) — otherwise restricting the draw would be vacuous."""
    for board_size in (19, 25):
        n_cells = board_size * board_size
        scatters = get_policy_scatters(board_size, has_pass=True)
        derived = [
            sym for sym in range(12)
            if len({int(x) for x in scatters[sym][:n_cells]}) == n_cells
        ]
        assert derived == list(WINDOW_PRESERVING_SYMS), (
            f"board_size={board_size}: derived window-preserving set {derived} disagrees with "
            f"WINDOW_PRESERVING_SYMS {list(WINDOW_PRESERVING_SYMS)}"
        )
        for sym in range(12):
            n_unique = len({int(x) for x in scatters[sym][:n_cells]})
            if sym in WINDOW_PRESERVING_SYMS:
                assert n_unique == n_cells, f"board_size={board_size}: sym {sym} inside the " \
                    "subgroup must be a bijection of the window"
            else:
                assert n_unique < n_cells, f"board_size={board_size}: sym {sym} outside the " \
                    "subgroup drops no cell — the restriction would be vacuous"


def test_draw_window_preserving_syms_support_is_exact() -> None:
    """A large seeded draw (numpy is autouse-reseeded per test, see tests/conftest.py) hits
    every element of {0,3,6,9} and NOTHING else — the support is exact, not merely a subset (an
    all-zero degenerate draw would pass a subset-only check for free)."""
    draws = draw_window_preserving_syms(4096)
    seen = {int(x) for x in draws}
    assert seen == set(WINDOW_PRESERVING_SYMS), f"draw support must be exactly " \
        f"{set(WINDOW_PRESERVING_SYMS)}, got {seen}"


# ── source-presence at both call sites (boundary pin, both directions) ──────────────────────
def test_batch_assembly_recent_rows_draw_is_gated() -> None:
    """`_augment_recent_rows` must call the GATED draw, not `np.random.randint(0, 12, ...)`
    and not the flat restricted draw — either reversion REDs this."""
    assert "draw_record_syms(" in _BATCH_ASSEMBLY_SRC, (
        "batch_assembly.py's RecentBuffer augmentation must draw via draw_record_syms"
    )
    assert "spread_mask(" in _BATCH_ASSEMBLY_SRC, (
        "batch_assembly.py must certify each row via spread_mask before drawing"
    )
    assert "randint(0, 12" not in _BATCH_ASSEMBLY_SRC, (
        "batch_assembly.py must not draw sym indices uniform over all 12 elements"
    )
    assert "draw_window_preserving_syms(" not in _BATCH_ASSEMBLY_SRC, (
        "batch_assembly.py must not restrict UNCONDITIONALLY — compact rows keep all 12"
    )


def test_pretrain_dataset_collate_draw_is_gated() -> None:
    """`make_augmented_collate`'s closure must call the GATED draw, not
    `np.random.randint(0, 12, ...)` and not the flat restricted draw."""
    assert "draw_record_syms(" in _PRETRAIN_DATASET_SRC, (
        "pretrain/dataset.py's collate must draw via draw_record_syms"
    )
    assert "spread_mask(" in _PRETRAIN_DATASET_SRC, (
        "pretrain/dataset.py must certify each row via spread_mask before drawing"
    )
    assert "randint(0, 12" not in _PRETRAIN_DATASET_SRC, (
        "pretrain/dataset.py must not draw sym indices uniform over all 12 elements"
    )
    assert "draw_window_preserving_syms(" not in _PRETRAIN_DATASET_SRC, (
        "pretrain/dataset.py must not restrict UNCONDITIONALLY — compact rows keep all 12"
    )


# ── R245(c): D, the dropped-cell set ────────────────────────────────────────────────────────
def _closed_form_dropped(board_size: int) -> np.ndarray:
    """D by the geometric argument: a dropping element's matrix has one `(±1, ±1)` row, so the
    surviving condition is `|q + r| <= half`. Independent of `get_dropped_cells`, which derives
    D by APPLYING the transform — the two must agree."""
    half = (board_size - 1) // 2
    flat = np.arange(board_size * board_size)
    q = flat // board_size - half
    r = flat % board_size - half
    return np.flatnonzero(np.abs(q + r) > half).astype(np.int64)


def test_dropped_cells_are_derived_not_transcribed() -> None:
    """`get_dropped_cells` matches an INDEPENDENT derivation at every shipped board size, and
    is non-empty — an empty D would make the whole gate vacuous (every row 'compact')."""
    for board_size in (19, 25):
        derived = get_dropped_cells(board_size)
        assert derived.size > 0, f"board_size={board_size}: D must be non-empty"
        assert np.array_equal(derived, _closed_form_dropped(board_size)), (
            f"board_size={board_size}: get_dropped_cells disagrees with the |q + r| > half "
            "derivation"
        )


def test_all_dropping_elements_drop_the_identical_cell_set() -> None:
    """The equivalence the BINARY gate rests on: all eight window-dropping elements delete the
    SAME source cells, so one flag per row suffices; the four preserving ones delete none."""
    for board_size in (19, 25):
        half = (board_size - 1) // 2
        flat = np.arange(board_size * board_size)
        q0 = flat // board_size - half
        r0 = flat % board_size - half
        expected = get_dropped_cells(board_size)
        n_dropping = 0
        for sym in range(12):
            q, r = (r0, q0) if sym >= 6 else (q0, r0)
            for _ in range(sym % 6):
                q, r = -r, q + r
            inside = (q + half >= 0) & (q + half < board_size) & (r + half >= 0) & (r + half < board_size)
            dropped = np.flatnonzero(~inside).astype(np.int64)
            if sym in WINDOW_PRESERVING_SYMS:
                assert dropped.size == 0, f"board_size={board_size}: preserving sym {sym} drops"
            else:
                n_dropping += 1
                assert np.array_equal(dropped, expected), (
                    f"board_size={board_size}: dropping sym {sym} deletes a DIFFERENT cell set "
                    "— the binary compact/spread gate would no longer be exact"
                )
        assert n_dropping == 12 - len(WINDOW_PRESERVING_SYMS)


# ── R245(c): the neutral-aware spread mask ──────────────────────────────────────────────────
def _interior_cell() -> int:
    d = set(int(x) for x in get_dropped_cells(_BOARD))
    return next(c for c in range(_N_CELLS) if c not in d)


def _blank_rows(n: int):
    """n all-NEUTRAL rows: states 0, chain 0, policy 0.0, ownership 1 (= empty), wl 0."""
    return (
        np.zeros((n, _N_PLANES, _BOARD, _BOARD), dtype=np.float16),
        np.zeros((n, 6, _BOARD, _BOARD), dtype=np.float16),
        np.zeros((n, _N_CELLS + 1), dtype=np.float32),
        np.ones((n, _N_CELLS), dtype=np.uint8),
        np.zeros((n, _N_CELLS), dtype=np.uint8),
    )


def test_spread_mask_flags_content_on_d_per_channel() -> None:
    """One row per channel, each carrying content ONLY on a dropped cell — every one must read
    SPREAD; the all-neutral row and the interior-content row must read COMPACT."""
    d = int(get_dropped_cells(_BOARD)[0])
    interior = _interior_cell()
    s, _c, p, own, wl = _blank_rows(6)
    # row 0: all neutral                      → compact
    # row 1: content at an interior cell only → compact
    s[1, 0, interior // _BOARD, interior % _BOARD] = 1.0
    p[1, interior] = 1.0
    own[1, interior] = 2
    wl[1, interior] = 1
    # rows 2..5: one channel each, on D       → spread
    s[2, 0, d // _BOARD, d % _BOARD] = 1.0
    p[3, d] = 1.0
    own[4, d] = 2
    wl[5, d] = 1

    mask = spread_mask(_BOARD, states=s, policies=p, ownership=own, winning_line=wl)
    assert mask.tolist() == [False, False, True, True, True, True]


def test_spread_mask_is_ownership_neutral_aware() -> None:
    """`ownership`'s neutral is 1 (= empty), NOT 0 — 0 means 'owned by P2'. A blanket
    all-channels-zero check would call BOTH rows below compact and then let a dropping element
    delete a real ownership label."""
    d = int(get_dropped_cells(_BOARD)[0])
    _s, _c, _p, own, _wl = _blank_rows(3)
    own[1, d] = 2  # P1 on a dropped cell
    own[2, d] = 0  # P2 on a dropped cell
    mask = spread_mask(_BOARD, ownership=own)
    assert mask.tolist() == [False, True, True]


def test_spread_mask_ignores_the_positionally_invariant_pass_slot() -> None:
    """The policy pass slot (index n_cells) rides through every element unchanged, so it can
    never be clipped and must not make a row spread."""
    _s, _c, p, _own, _wl = _blank_rows(1)
    p[0, _N_CELLS] = 1.0
    assert spread_mask(_BOARD, policies=p).tolist() == [False]


# ── R245(c): the gated draw ─────────────────────────────────────────────────────────────────
def test_draw_record_syms_support_is_exact_on_both_arms() -> None:
    """Support-exact in BOTH directions on both arms: compact rows recover all 12, spread rows
    see nothing outside {0,3,6,9}. A subset-only check would pass for a degenerate draw."""
    compact_draws = draw_record_syms(np.zeros(4096, dtype=bool))
    assert {int(x) for x in compact_draws} == set(range(12))

    spread_draws = draw_record_syms(np.ones(4096, dtype=bool))
    assert {int(x) for x in spread_draws} == set(WINDOW_PRESERVING_SYMS)


def test_draw_record_syms_gates_row_by_row_in_one_call() -> None:
    """The gate is PER ROW inside a single mixed call, not per call."""
    mask = np.array([True, False] * 2048, dtype=bool)
    draws = draw_record_syms(mask)
    assert {int(x) for x in draws[mask]} == set(WINDOW_PRESERVING_SYMS)
    assert {int(x) for x in draws[~mask]} == set(range(12))


# ── R245(c): the two sites' BEHAVIOUR (no clipped copy reaches a batch) ──────────────────────
_PROBE_CONTENT_CELLS = 5


def _probe_rows(n: int, *, spread: bool):
    """`n` identical probe rows.

    The policy carries a DISTINCT value per cell, so the 12 per-sym scatters of it are pairwise
    distinguishable and an emitted row identifies the element that produced it. `spread=False`
    leaves every cell of D neutral (window-fitting); `spread=True` puts content on D.
    """
    d = get_dropped_cells(_BOARD)
    d_set = set(int(x) for x in d)
    s, c, p, own, wl = _blank_rows(n)

    policy_cells = range(_N_CELLS) if spread else (c_ for c_ in range(_N_CELLS) if c_ not in d_set)
    for cell in policy_cells:
        p[:, cell] = float(cell + 1)
    p[:, _N_CELLS] = 0.5  # pass slot

    content = [_interior_cell() + k for k in range(_PROBE_CONTENT_CELLS)]
    if spread:
        content.append(int(d[0]))
    for cell in content:
        s[:, 0, cell // _BOARD, cell % _BOARD] = 1.0
        own[:, cell] = 2
        wl[:, cell] = 1
    return s, c, p, own, wl


def _expected_policy_scatters(policy_row: np.ndarray) -> list[np.ndarray]:
    """The 12 per-sym expected policy vectors, built with the SAME LUTs the sites scatter with.
    Asserted pairwise-distinct so a match identifies the drawn element uniquely."""
    luts = get_policy_scatters(_BOARD, has_pass=True)
    expected = [policy_row[lut] for lut in luts]
    for a in range(12):
        for b in range(a + 1, 12):
            assert not np.array_equal(expected[a], expected[b]), (
                f"test setup: syms {a} and {b} must be distinguishable"
            )
    return expected


def _identify_syms(emitted: np.ndarray, expected: list[np.ndarray]) -> set[int]:
    """The set of elements observed across `emitted` rows. Each row must match EXACTLY one of
    the 12 expectations — a clipped copy matches the dropping element's clipped expectation and
    is therefore visible here, not silently accepted."""
    seen: set[int] = set()
    for row in emitted:
        matches = [s for s in range(12) if np.array_equal(expected[s], row)]
        assert len(matches) == 1, f"row matched {len(matches)} elements, expected exactly 1"
        seen.add(matches[0])
    return seen


def _mass(states: np.ndarray, policies: np.ndarray) -> list[tuple[int, float]]:
    """Per-row (nonzero state cells, policy sum). Small exact integers, so the f32 sum is exact
    and a clipped copy — which loses cells — cannot alias onto a conserved one."""
    n = states.shape[0]
    return [
        (int(np.count_nonzero(states[i])), float(np.asarray(policies[i], dtype=np.float64).sum()))
        for i in range(n)
    ]


def _run_recent_rows(spread: bool, n: int):
    from mantis.encoding import lookup as _lookup_encoding
    from mantis.encoding import opp_stone_slot
    from mantis.train.batch_assembly import _augment_recent_rows

    opp_slot = opp_stone_slot(_lookup_encoding("v6"))
    s, c, p, own, wl = _probe_rows(n, spread=spread)
    s_out, _c_out, p_out, own_out, wl_out = _augment_recent_rows(
        s, c, p, own, wl, True, opp_slot
    )
    return (s, p, own, wl), (s_out, p_out, own_out, wl_out)


def test_augment_recent_rows_spread_rows_only_get_preserving_elements() -> None:
    """Operator pin 1 at the RecentBuffer site: rows a dropping element would clip see ONLY
    {0,3,6,9}, and every emitted row conserves its source's per-channel mass."""
    n = 192
    (s_in, p_in, own_in, wl_in), (s_out, p_out, own_out, wl_out) = _run_recent_rows(True, n)

    assert _identify_syms(p_out, _expected_policy_scatters(p_in[0])) == set(WINDOW_PRESERVING_SYMS)
    assert _mass(s_out, p_out) == _mass(s_in, p_in)
    assert int(np.count_nonzero(own_out != 1)) == int(np.count_nonzero(own_in != 1))
    assert int(np.count_nonzero(wl_out)) == int(np.count_nonzero(wl_in))


def test_augment_recent_rows_compact_rows_recover_the_full_group() -> None:
    """Operator pin 2 at the RecentBuffer site: window-fitting rows see all 12 elements, and
    mass is still conserved — the dropping elements delete only neutral padding here, and the
    numpy LUT's `n_cells - 1` corner fallback IS that neutral (the corner is in D)."""
    n = 192
    (s_in, p_in, own_in, wl_in), (s_out, p_out, own_out, wl_out) = _run_recent_rows(False, n)

    assert _identify_syms(p_out, _expected_policy_scatters(p_in[0])) == set(range(12))
    assert _mass(s_out, p_out) == _mass(s_in, p_in)
    assert int(np.count_nonzero(own_out != 1)) == int(np.count_nonzero(own_in != 1))
    assert int(np.count_nonzero(wl_out)) == int(np.count_nonzero(wl_in))


def _run_collate(spread: bool, n: int):
    collate = pretrain_dataset.make_augmented_collate(True, "v6")
    s, _c, p, _own, _wl = _probe_rows(n, spread=spread)
    batch = [(s[i], p[i], 0.0) for i in range(n)]
    out = collate(batch)
    return (s, p), (out[0].numpy(), out[2].numpy())


def test_make_augmented_collate_spread_rows_only_get_preserving_elements() -> None:
    """Operator pin 1 at the pretrain-collate site."""
    n = 192
    (s_in, p_in), (s_out, p_out) = _run_collate(True, n)
    assert _identify_syms(p_out, _expected_policy_scatters(p_in[0])) == set(WINDOW_PRESERVING_SYMS)
    assert _mass(s_out, p_out) == _mass(s_in, p_in)


def test_make_augmented_collate_compact_rows_recover_the_full_group() -> None:
    """Operator pin 2 at the pretrain-collate site."""
    n = 192
    (s_in, p_in), (s_out, p_out) = _run_collate(False, n)
    assert _identify_syms(p_out, _expected_policy_scatters(p_in[0])) == set(range(12))
    assert _mass(s_out, p_out) == _mass(s_in, p_in)


def test_the_mass_metric_detects_a_clipped_copy() -> None:
    """Anti-vacuity for the mass assertions above: scattering a SPREAD row by hand under a
    dropping element DOES lose mass. If it did not, those assertions would pass for free."""
    dropping = next(s for s in range(12) if s not in WINDOW_PRESERVING_SYMS)
    lut = get_policy_scatters(_BOARD, has_pass=True)[dropping]
    s_in, _c, p_in, _own, _wl = _probe_rows(1, spread=True)

    clipped_policy = p_in[0][lut]
    assert float(clipped_policy.sum()) != float(p_in[0].sum()), (
        f"sym {dropping} on a spread row must change the policy mass"
    )

    spatial = s_in[0].reshape(_N_PLANES, _N_CELLS)
    clipped_state = spatial[:, lut[:_N_CELLS]]
    assert int(np.count_nonzero(clipped_state)) != int(np.count_nonzero(s_in[0])), (
        f"sym {dropping} on a spread row must change the nonzero state-cell count"
    )


# ── R245(c): each SITE must certify every channel it scatters (per-channel isolation) ────────
#
# The probe rows above put D content in EVERY channel at once, so dropping a channel from a
# site's `spread_mask(...)` argument list never changes their mask and the whole suite stays
# green. The rows below are spread via EXACTLY ONE channel, so a site that stops certifying
# that channel mis-classifies them as compact, draws a dropping element, and loses mass.
#
# This is not theoretical for the policy arm: a bootstrap pretrain triple's policy target is
# the NEXT move, which by definition sits on an EMPTY cell — states is neutral there while
# policies is not. A states-only certification would clip that row's policy target.
_ISOLATION_CHANNELS = ("states", "policies", "ownership", "winning_line")


def _isolated_spread_rows(n: int, channel: str):
    """`n` identical rows whose ONLY D-content lives in `channel`.

    The policy always carries a distinct value per NON-D cell, so an emitted row still
    identifies the element that produced it (`_expected_policy_scatters` asserts the 12
    scatters stay pairwise distinct). For `channel == "policies"` the D cells carry values too
    — that IS the isolated content; for every other channel the policy stays neutral on D.
    """
    d = get_dropped_cells(_BOARD)
    d_set = {int(x) for x in d}
    s, c, p, own, wl = _blank_rows(n)

    for cell in range(_N_CELLS):
        if cell in d_set and channel != "policies":
            continue
        p[:, cell] = float(cell + 1)
    p[:, _N_CELLS] = 0.5  # pass slot — positionally invariant, never in D

    target = int(d[0])
    if channel == "states":
        s[:, 0, target // _BOARD, target % _BOARD] = 1.0
    elif channel == "ownership":
        own[:, target] = 2
    elif channel == "winning_line":
        wl[:, target] = 1
    elif channel != "policies":
        raise AssertionError(f"unknown isolation channel {channel!r}")
    return s, c, p, own, wl


def _assert_isolation(channel: str, s, p, own, wl) -> None:
    """The load-bearing setup check: the row is spread via `channel` and via NOTHING else, so
    a site that omits `channel` from its certification call WILL classify it compact."""
    per_channel = {"states": s, "policies": p, "ownership": own, "winning_line": wl}
    assert spread_mask(_BOARD, **{channel: per_channel[channel]}).all(), (
        f"test setup: {channel} alone must make the row spread"
    )
    for other in _ISOLATION_CHANNELS:
        if other == channel:
            continue
        assert not spread_mask(_BOARD, **{other: per_channel[other]}).any(), (
            f"test setup: {other} must be neutral on D for the {channel}-isolated row"
        )


@pytest.mark.parametrize("channel", _ISOLATION_CHANNELS)
def test_augment_recent_rows_certifies_every_channel_it_scatters(channel: str) -> None:
    """`_augment_recent_rows` scatters states, policies, ownership AND winning_line, so it must
    certify all four. Dropping any one from its `spread_mask` call REDs this."""
    from mantis.encoding import lookup as _lookup_encoding
    from mantis.encoding import opp_stone_slot
    from mantis.train.batch_assembly import _augment_recent_rows

    n = 192
    s, c, p, own, wl = _isolated_spread_rows(n, channel)
    _assert_isolation(channel, s, p, own, wl)

    opp_slot = opp_stone_slot(_lookup_encoding("v6"))
    s_out, _c_out, p_out, own_out, wl_out = _augment_recent_rows(s, c, p, own, wl, True, opp_slot)

    assert _identify_syms(p_out, _expected_policy_scatters(p[0])) == set(WINDOW_PRESERVING_SYMS)
    assert _mass(s_out, p_out) == _mass(s, p)
    assert int(np.count_nonzero(own_out != 1)) == int(np.count_nonzero(own != 1))
    assert int(np.count_nonzero(wl_out)) == int(np.count_nonzero(wl))


@pytest.mark.parametrize("channel", ("states", "policies"))
def test_make_augmented_collate_certifies_every_channel_it_scatters(channel: str) -> None:
    """The pretrain collate scatters states and policies (chain is recomputed from the
    augmented stones, ownership/winning_line are not part of a pretrain triple), so it must
    certify exactly those two. Dropping either from its `spread_mask` call REDs this."""
    n = 192
    s, _c, p, own, wl = _isolated_spread_rows(n, channel)
    _assert_isolation(channel, s, p, own, wl)

    collate = pretrain_dataset.make_augmented_collate(True, "v6")
    out = collate([(s[i], p[i], 0.0) for i in range(n)])
    s_out, p_out = out[0].numpy(), out[2].numpy()

    assert _identify_syms(p_out, _expected_policy_scatters(p[0])) == set(WINDOW_PRESERVING_SYMS)
    assert _mass(s_out, p_out) == _mass(s, p)


# ── R245(c): the empty batch stays a no-op, not a crash ─────────────────────────────────────
def test_spread_mask_handles_an_empty_batch() -> None:
    """`n == 0` returns an empty mask. Every channel arm reshapes with an inferred `-1`, which
    numpy refuses to resolve at size 0, so without the early return all four arms raise —
    a narrowing of a function the flat draws it replaced handled cleanly."""
    empty = {
        "states": np.zeros((0, _N_PLANES, _BOARD, _BOARD), dtype=np.float16),
        "policies": np.zeros((0, _N_CELLS + 1), dtype=np.float32),
        "ownership": np.ones((0, _N_CELLS), dtype=np.uint8),
        "winning_line": np.zeros((0, _N_CELLS), dtype=np.uint8),
    }
    for name, arr in empty.items():
        mask = spread_mask(_BOARD, **{name: arr})
        assert mask.shape == (0,) and mask.dtype == np.bool_, f"{name} arm at n=0"
    all_arms = spread_mask(_BOARD, **empty)
    assert all_arms.shape == (0,) and all_arms.dtype == np.bool_
    assert draw_record_syms(all_arms).shape == (0,)


def test_spread_mask_still_rejects_a_call_with_no_channels() -> None:
    """The n == 0 early return must not swallow the genuine misuse error — a caller that
    supplies NO channel is asking for a certification of nothing and still raises."""
    with pytest.raises(ValueError):
        spread_mask(_BOARD)


def test_augment_recent_rows_handles_an_empty_batch() -> None:
    """The site inherits the empty-batch no-op end to end (no production path reaches n == 0
    today — `RecentBuffer.sample` never returns zero rows — but the site must not be the thing
    that breaks if one ever does)."""
    from mantis.encoding import lookup as _lookup_encoding
    from mantis.encoding import opp_stone_slot
    from mantis.train.batch_assembly import _augment_recent_rows

    s, c, p, own, wl = _blank_rows(0)
    out = _augment_recent_rows(s, c, p, own, wl, True, opp_stone_slot(_lookup_encoding("v6")))
    assert [int(x.shape[0]) for x in out] == [0, 0, 0, 0, 0]
