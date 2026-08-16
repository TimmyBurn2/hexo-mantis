//! R245(c) — the per-record losslessness gate on the DENSE replay arm.
//!
//! R8: over the 300-line soft cap by design. The gate is one mechanism with two
//! halves that are only meaningful together — the `compact` flag must be correct
//! at EVERY slot writer (six of them: the three pushes, the test helper, HEXB
//! load, and `resize`'s linearise), and the sample path must then draw the group
//! that flag licenses. Splitting the writer half from the draw half would leave
//! either file able to pass while the mechanism is broken end to end: a writer
//! suite with no sampler proves nothing about training data, and a sampler suite
//! with hand-set flags proves nothing about the writers. The shared row builders
//! (compact / spread / ownership-only-spread) and the derived per-sym expectation
//! table are the common vocabulary both halves are written in.
//!
//! Operator pins under test:
//!   1. a SPREAD record (one a symmetry element would clip) gets ONLY {0,3,6,9};
//!   2. a COMPACT (window-fitting) record gets the full 12;
//!   3. NO clipped copy is ever trained — checked as per-channel mass
//!      conservation against the source slot, over a MIXED buffer.

use std::collections::HashSet;

use half::f16;
use rand::rngs::StdRng;
use rand::SeedableRng;

use mantis_encoding::registry::lookup_or_panic;
use mantis_selfplay::replay::push_config::{PushGameConfig, PushManyConfig, PushSingleConfig};
use mantis_selfplay::replay::sample::{ApplySymDst, ApplySymSlices, ApplySymSrc};
use mantis_selfplay::replay::sym::{N_SYMS, WINDOW_PRESERVING_SYMS};
use mantis_selfplay::replay::ReplayBuffer;

const ENC: &str = "v6";

/// (state, chain, policy, aux) strides + `n_cells` for the encoding under test.
fn shape() -> (usize, usize, usize, usize, usize) {
    let s = lookup_or_panic(ENC);
    (s.state_stride(), s.chain_stride(), s.policy_stride(), s.aux_stride(), s.n_cells())
}

/// One dense record in push-config form.
#[derive(Clone)]
struct Row {
    state: Vec<f16>,
    chain: Vec<f16>,
    policy: Vec<f32>,
    own: Vec<u8>,
    wl: Vec<u8>,
}

impl Row {
    /// An all-NEUTRAL row: states 0, chain 0, policy 0.0, ownership 1 (= empty),
    /// `winning_line` 0 — the same neutrals `ReplayBuffer::build` initialises with.
    fn neutral() -> Self {
        let (st, ch, po, ax, _) = shape();
        Row {
            state: vec![f16::from_f32(0.0); st],
            chain: vec![f16::from_f32(0.0); ch],
            policy: vec![0.0f32; po],
            own: vec![1u8; ax],
            wl: vec![0u8; ax],
        }
    }

    /// Write non-neutral content into every channel at `cell`.
    fn with_content_at(mut self, cell: usize) -> Self {
        let (_, _, po, ax, n_cells) = shape();
        self.state[cell] = f16::from_f32(1.0);
        self.chain[cell] = f16::from_f32(0.5);
        self.policy[cell] = 1.0;
        if po > n_cells {
            self.policy[n_cells] = 0.25; // pass slot — positionally invariant
        }
        self.own[cell % ax] = 2; // P1 (neutral is 1)
        self.wl[cell % ax] = 1;
        self
    }

    fn push_config(&self, outcome: f32) -> PushSingleConfig<'_> {
        PushSingleConfig {
            state: &self.state,
            chain_planes: &self.chain,
            policy: &self.policy,
            outcome,
            ownership: &self.own,
            winning_line: &self.wl,
            game_id: -1,
            game_length: 0,
            is_full_search: true,
            position_index: 0,
            value_target_valid: true,
        }
    }
}

/// The dropped-cell set D and one interior (never-dropped) cell, both DERIVED
/// from the buffer's own sym tables.
fn dropped_and_interior(buf: &ReplayBuffer) -> (usize, usize) {
    let dropped = &buf.sym_tables.dropped_cells;
    assert!(!dropped.is_empty(), "D must be non-empty or every test here is vacuous");
    let d: HashSet<usize> = dropped.iter().map(|&c| c as usize).collect();
    let interior = (0..buf.sym_tables.n_cells)
        .find(|c| !d.contains(c))
        .expect("the window must have at least one never-dropped cell");
    (dropped[0] as usize, interior)
}

/// A record that fits the window under every element (content only at `interior`).
fn compact_row(interior: usize) -> Row {
    Row::neutral().with_content_at(interior)
}

/// A record a dropping element would clip (content at a cell of D).
fn spread_row(dropped_cell: usize) -> Row {
    Row::neutral().with_content_at(dropped_cell)
}

// ── flag correctness at every slot writer ──────────────────────────────────────

#[test]
fn push_impl_flags_compact_and_spread_rows() {
    let mut buf = ReplayBuffer::new(4, ENC);
    let (d, interior) = dropped_and_interior(&buf);

    buf.push_impl(compact_row(interior).push_config(0.0)).unwrap();
    buf.push_impl(spread_row(d).push_config(1.0)).unwrap();

    assert_eq!(buf.compact_at(0), 1, "a window-fitting record must be flagged COMPACT");
    assert_eq!(buf.compact_at(1), 0, "a record with content on D must be flagged SPREAD");
}

#[test]
fn push_game_impl_flags_each_row_independently() {
    let (st, ch, po, ax, _) = shape();
    let mut buf = ReplayBuffer::new(4, ENC);
    let (d, interior) = dropped_and_interior(&buf);

    // Rows in order: compact, spread, compact.
    let rows = [compact_row(interior), spread_row(d), compact_row(interior)];
    let mut states = Vec::with_capacity(3 * st);
    let mut chains = Vec::with_capacity(3 * ch);
    let mut policies = Vec::with_capacity(3 * po);
    let mut owns = Vec::with_capacity(3 * ax);
    let mut wls = Vec::with_capacity(3 * ax);
    for r in &rows {
        states.extend_from_slice(&r.state);
        chains.extend_from_slice(&r.chain);
        policies.extend_from_slice(&r.policy);
        owns.extend_from_slice(&r.own);
        wls.extend_from_slice(&r.wl);
    }

    buf.push_game_impl(PushGameConfig {
        states: &states,
        chain_planes: &chains,
        policies: &policies,
        outcomes: &[0.0, 1.0, 2.0],
        ownership: &owns,
        winning_line: &wls,
        game_id: 7,
        game_length: 0,
        is_full_search: None,
        position_indices: None,
        value_target_valid: None,
    })
    .unwrap();

    assert_eq!(
        [buf.compact_at(0), buf.compact_at(1), buf.compact_at(2)],
        [1u8, 0, 1],
        "push_game_impl must flag every row from the row it wrote, not the batch"
    );
}

#[test]
fn push_many_impl_flags_each_row_independently() {
    let (st, ch, po, ax, _) = shape();
    let mut buf = ReplayBuffer::new(4, ENC);
    let (d, interior) = dropped_and_interior(&buf);

    let rows = [spread_row(d), compact_row(interior)];
    let mut states = Vec::with_capacity(2 * st);
    let mut chains = Vec::with_capacity(2 * ch);
    let mut policies = Vec::with_capacity(2 * po);
    let mut owns = Vec::with_capacity(2 * ax);
    let mut wls = Vec::with_capacity(2 * ax);
    for r in &rows {
        states.extend_from_slice(&r.state);
        chains.extend_from_slice(&r.chain);
        policies.extend_from_slice(&r.policy);
        owns.extend_from_slice(&r.own);
        wls.extend_from_slice(&r.wl);
    }

    buf.push_many_impl(PushManyConfig {
        states: &states,
        chain_planes: &chains,
        policies: &policies,
        outcomes: &[0.0, 1.0],
        ownership: &owns,
        winning_line: &wls,
        game_lengths: &[0, 0],
        is_full_search: &[1, 1],
        position_indices: None,
        value_target_valid: None,
    })
    .unwrap();

    assert_eq!([buf.compact_at(0), buf.compact_at(1)], [0u8, 1]);
}

#[test]
fn push_for_test_rows_are_compact() {
    let mut buf = ReplayBuffer::new(4, ENC);
    buf.push_for_test(1.0, 10, true);
    assert_eq!(
        buf.compact_at(0),
        1,
        "push_for_test writes an all-neutral row, which is compact by construction"
    );
}

/// The NEUTRAL-AWARENESS witness. `ownership`'s neutral is 1 (= empty), not 0 —
/// a blanket "all channels zero on D" test would call this row compact, and a
/// dropping element would then silently delete a real P1 ownership label.
#[test]
fn ownership_only_spread_row_is_classified_spread() {
    let mut buf = ReplayBuffer::new(4, ENC);
    let (d, _) = dropped_and_interior(&buf);

    let mut row = Row::neutral();
    row.own[d] = 2; // P1 on a dropped cell; every other channel stays neutral

    buf.push_impl(row.push_config(0.0)).unwrap();
    assert_eq!(
        buf.compact_at(0),
        0,
        "ownership content on D must read as SPREAD (neutral is 1, not 0)"
    );

    // Anti-vacuity: the SAME row with ownership neutral (1) is compact, so the
    // classification above is driven by the ownership byte and nothing else.
    let mut buf2 = ReplayBuffer::new(4, ENC);
    let clean = Row::neutral();
    buf2.push_impl(clean.push_config(0.0)).unwrap();
    assert_eq!(buf2.compact_at(0), 1);
}

/// An ownership byte of 0 on D means "owned by P2" — also content, also spread.
#[test]
fn ownership_zero_on_d_is_also_spread() {
    let mut buf = ReplayBuffer::new(4, ENC);
    let (d, _) = dropped_and_interior(&buf);
    let mut row = Row::neutral();
    row.own[d] = 0; // P2
    buf.push_impl(row.push_config(0.0)).unwrap();
    assert_eq!(buf.compact_at(0), 0);
}

/// The channel this row's single non-neutral value goes into.
#[derive(Clone, Copy)]
enum Channel {
    State,
    Chain,
    Policy,
    Ownership,
    WinningLine,
}

/// Write ONE non-neutral value into `channel` at plane-base `base` + `cell`, leaving every
/// other channel — and every other cell — neutral.
fn write_one(row: &mut Row, channel: Channel, base: usize, cell: usize) {
    match channel {
        Channel::State => row.state[base + cell] = f16::from_f32(1.0),
        Channel::Chain => row.chain[base + cell] = f16::from_f32(0.5),
        Channel::Policy => row.policy[base + cell] = 1.0,
        Channel::Ownership => row.own[base + cell] = 2,
        Channel::WinningLine => row.wl[base + cell] = 1,
    }
}

/// Per-channel, per-plane ISOLATION rows — the mutation self-test for every INPUT of
/// `slot_is_compact` (LAW-07).
///
/// Every other row builder in this file writes all five channels at the SAME cell and only
/// into plane 0, so any one surviving channel check classifies those rows correctly and the
/// other four are redundant TO THE TEST: deleting the `winning_line` check, or either
/// multi-plane scan, or bounding either scan to plane 0, leaves the whole suite green. The
/// rows below carry their D content in exactly ONE channel, and for the two multi-plane
/// channels in the LAST plane as well as plane 0, so each of those deletions REDs here.
///
/// Plane counts are DERIVED from the strides (`state_stride / n_cells`,
/// `chain_stride / n_cells`), never transcribed, and the >1-plane premise is asserted so the
/// deep-plane arms cannot go vacuous under a future encoding.
#[test]
fn each_channel_alone_on_a_dropped_cell_reads_spread() {
    let (state_stride, chain_stride, _po, _ax, n_cells) = shape();
    let n_state_planes = state_stride / n_cells;
    let n_chain_planes = chain_stride / n_cells;
    assert!(
        n_state_planes > 1 && n_chain_planes > 1,
        "test setup: encoding {ENC} must carry more than one plane on BOTH states and chain, \
         or the deep-plane arms below are vacuous (state planes {n_state_planes}, chain \
         planes {n_chain_planes})"
    );
    let last_state_plane = (n_state_planes - 1) * n_cells;
    let last_chain_plane = (n_chain_planes - 1) * n_cells;

    let geometry = ReplayBuffer::new(1, ENC);
    let (d, interior) = dropped_and_interior(&geometry);
    drop(geometry);

    let cases: [(&str, Channel, usize); 7] = [
        ("state plane 0", Channel::State, 0),
        ("state LAST plane", Channel::State, last_state_plane),
        ("chain plane 0", Channel::Chain, 0),
        ("chain LAST plane", Channel::Chain, last_chain_plane),
        ("policy", Channel::Policy, 0),
        ("ownership", Channel::Ownership, 0),
        ("winning_line", Channel::WinningLine, 0),
    ];

    for (label, channel, base) in cases {
        // SPREAD: the row's ONE non-neutral value sits on a dropped cell.
        let mut spread = Row::neutral();
        write_one(&mut spread, channel, base, d);
        let mut buf = ReplayBuffer::new(1, ENC);
        buf.push_impl(spread.push_config(0.0)).unwrap();
        assert_eq!(
            buf.compact_at(0),
            0,
            "{label} alone on a dropped cell must read SPREAD — `slot_is_compact` is not \
             consulting that channel/plane"
        );

        // Anti-vacuity: the SAME single write at an INTERIOR cell must read COMPACT, so the
        // verdict above is driven by the CELL, not merely by the channel being touched at all.
        let mut compact = Row::neutral();
        write_one(&mut compact, channel, base, interior);
        let mut buf2 = ReplayBuffer::new(1, ENC);
        buf2.push_impl(compact.push_config(0.0)).unwrap();
        assert_eq!(
            buf2.compact_at(0),
            1,
            "{label} alone at an interior cell must read COMPACT"
        );
    }
}

#[test]
fn ring_wraparound_overwrite_recomputes_the_flag() {
    let mut buf = ReplayBuffer::new(2, ENC);
    let (d, interior) = dropped_and_interior(&buf);

    buf.push_impl(compact_row(interior).push_config(0.0)).unwrap();
    buf.push_impl(compact_row(interior).push_config(1.0)).unwrap();
    assert_eq!([buf.compact_at(0), buf.compact_at(1)], [1u8, 1]);

    // Wrap: slot 0 is overwritten by a spread record.
    buf.push_impl(spread_row(d).push_config(2.0)).unwrap();
    assert_eq!(
        buf.compact_at(0),
        0,
        "an overwritten slot must take the NEW row's flag, never keep the old one"
    );
    assert_eq!(buf.compact_at(1), 1);
}

#[test]
fn resize_carries_the_flag_through_linearise() {
    let mut buf = ReplayBuffer::new(3, ENC);
    let (d, interior) = dropped_and_interior(&buf);

    // Ring order after 4 pushes into capacity 3: head = 1, slots = [r3, r1, r2].
    for (i, row) in [
        compact_row(interior),
        compact_row(interior),
        spread_row(d),
        spread_row(d),
    ]
    .iter()
    .enumerate()
    {
        buf.push_impl(row.push_config(i as f32)).unwrap();
    }
    assert_eq!(buf.head, 1, "test setup: the ring must have wrapped");
    assert_eq!([buf.compact_at(0), buf.compact_at(1), buf.compact_at(2)], [0u8, 1, 0]);

    buf.resize(6).unwrap();

    // Linearised order (oldest → newest) is r1, r2, r3 = compact, spread, spread.
    // Not-rotating the flag column would leave [0, 1, 0] against that data.
    assert_eq!([buf.compact_at(0), buf.compact_at(1), buf.compact_at(2)], [1u8, 0, 0]);
    for slot in 0..buf.size() {
        assert_eq!(
            buf.compact_at(slot),
            u8::from(buf.slot_is_compact(slot)),
            "slot {slot}: the stored flag must still describe the row sitting there"
        );
    }
    // The grown tail defaults to SPREAD (uncertified), never to compact.
    for slot in buf.size()..buf.capacity() {
        assert_eq!(buf.compact_at(slot), 0, "grown slot {slot} must default to spread");
    }
}

#[test]
fn hexb_roundtrip_recomputes_flags_for_a_mixed_buffer() {
    let path = std::env::temp_dir().join(format!(
        "r245c_mixed_{}_{}.hexb",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    ));

    let mut writer = ReplayBuffer::new(4, ENC);
    let (d, interior) = dropped_and_interior(&writer);
    writer.push_impl(spread_row(d).push_config(0.0)).unwrap();
    writer.push_impl(compact_row(interior).push_config(1.0)).unwrap();
    writer.push_impl(spread_row(d).push_config(2.0)).unwrap();
    writer.save_to_path(path.to_str().unwrap()).unwrap();

    let mut reader = ReplayBuffer::new(4, ENC);
    // A fresh buffer's flag column is all-zero, so the compact row's 0 → 1
    // transition below is only reachable if the loader actually recomputes.
    assert!(reader.compact.iter().all(|&c| c == 0));
    let n = reader.load_from_path(path.to_str().unwrap()).unwrap();
    assert_eq!(n, 3);

    assert_eq!(
        [reader.compact_at(0), reader.compact_at(1), reader.compact_at(2)],
        [0u8, 1, 0],
        "the HEXB load path must recompute the flag per loaded row (it is not on the wire)"
    );
    for slot in 0..n {
        assert_eq!(reader.compact_at(slot), u8::from(reader.slot_is_compact(slot)));
    }

    let _ = std::fs::remove_file(path);
}

// ── the gated draw, observed through the sample cores ──────────────────────────

/// The 12 per-sym expected policy outputs for `row`, produced by the SAME
/// `apply_sym` kernel the sample path uses over neutral-initialised destinations.
/// Asserted pairwise-distinct so a match identifies the drawn element uniquely.
fn expected_policies(buf: &ReplayBuffer, row: &Row) -> Vec<Vec<f32>> {
    let (st, ch, po, ax, _) = shape();
    let mut out = Vec::with_capacity(N_SYMS);
    for sym in 0..N_SYMS {
        let mut dst_state = vec![0u16; st];
        let mut dst_chain = vec![0u16; ch];
        let mut dst_pol = vec![0.0f32; po];
        let mut dst_own = vec![1u8; ax];
        let mut dst_wl = vec![0u8; ax];
        let src_state: Vec<u16> = row.state.iter().map(|v| v.to_bits()).collect();
        let src_chain: Vec<u16> = row.chain.iter().map(|v| v.to_bits()).collect();
        ReplayBuffer::apply_sym(
            sym,
            ApplySymSlices {
                src: ApplySymSrc {
                    state: &src_state,
                    chain: &src_chain,
                    policy: &row.policy,
                    own: &row.own,
                    wl: &row.wl,
                },
                dst: ApplySymDst {
                    state: &mut dst_state,
                    chain: &mut dst_chain,
                    policy: &mut dst_pol,
                    own: &mut dst_own,
                    wl: &mut dst_wl,
                },
                tables: buf.sym_tables,
            },
        );
        out.push(dst_pol);
    }
    for a in 0..N_SYMS {
        for b in (a + 1)..N_SYMS {
            assert_ne!(
                out[a], out[b],
                "test setup: syms {a} and {b} must produce distinguishable outputs"
            );
        }
    }
    out
}

/// A row whose policy separates all 12 elements: a distinct value per cell.
fn discriminating_row(cells: impl Iterator<Item = usize>) -> Row {
    let (_, _, po, _, n_cells) = shape();
    let mut row = Row::neutral();
    for cell in cells {
        row.policy[cell] = (cell + 1) as f32;
    }
    if po > n_cells {
        row.policy[n_cells] = 0.5;
    }
    row
}

fn observed_syms(buf: &mut ReplayBuffer, expected: &[Vec<f32>], draws: usize) -> HashSet<usize> {
    let (_, _, po, _, _) = shape();
    let mut seen = HashSet::new();
    for _ in 0..draws {
        let out = buf.sample_batch_core(1, true).unwrap();
        let emitted = &out.policies[..po];
        let matches: Vec<usize> =
            (0..N_SYMS).filter(|&s| expected[s].as_slice() == emitted).collect();
        assert_eq!(
            matches.len(),
            1,
            "an emitted record must equal the exact transform of its source under \
             exactly one D6 element (got {} matches)",
            matches.len()
        );
        seen.insert(matches[0]);
    }
    seen
}

#[test]
fn a_compact_only_buffer_draws_the_full_group() {
    let mut buf = ReplayBuffer::new(1, ENC);
    buf.rng = StdRng::seed_from_u64(0xC011_AC70);
    let dropped: HashSet<usize> =
        buf.sym_tables.dropped_cells.iter().map(|&c| c as usize).collect();
    let n_cells = buf.sym_tables.n_cells;
    let row = discriminating_row((0..n_cells).filter(|c| !dropped.contains(c)));

    buf.push_impl(row.push_config(0.0)).unwrap();
    assert_eq!(buf.compact_at(0), 1, "test setup: the row must be compact");

    let expected = expected_policies(&buf, &row);
    let seen = observed_syms(&mut buf, &expected, 512);
    assert_eq!(
        seen,
        (0..N_SYMS).collect::<HashSet<usize>>(),
        "a compact record must recover the FULL 12-element group"
    );
    // R266/F-P1/N1 — the LAW-18 fire-rate counter, driven by the SAME 512 draws above.
    // MUTATION THAT REDS IT: delete the `record_symmetry_draw` call at either
    // `sample.rs` call site (the counter goes dark while the draw keeps drawing), or
    // swap the `compact`/`spread` counters (a compact-only buffer would then read
    // `spread_draws() == 512`).
    assert_eq!(buf.compact_draws(), 512, "every draw from an all-compact buffer must tick compact_draws");
    assert_eq!(buf.spread_draws(), 0, "…and never spread_draws");
}

#[test]
fn a_spread_only_buffer_draws_exactly_the_window_preserving_subgroup() {
    let mut buf = ReplayBuffer::new(1, ENC);
    buf.rng = StdRng::seed_from_u64(0x5BEA_D100);
    let n_cells = buf.sym_tables.n_cells;
    // Content on EVERY cell, D included → the record is spread.
    let row = discriminating_row(0..n_cells);

    buf.push_impl(row.push_config(0.0)).unwrap();
    assert_eq!(buf.compact_at(0), 0, "test setup: the row must be spread");

    let expected = expected_policies(&buf, &row);
    let seen = observed_syms(&mut buf, &expected, 512);
    assert_eq!(
        seen,
        WINDOW_PRESERVING_SYMS.iter().copied().collect::<HashSet<usize>>(),
        "a spread record must draw EXACTLY the window-preserving subgroup — support \
         exact in both directions (nothing missing, nothing extra)"
    );
    // R266/F-P1/N1 — the LAW-18 fire-rate counter, driven by the SAME 512 draws above.
    // MUTATION THAT REDS IT: same as the compact-only twin above, inverted.
    assert_eq!(buf.spread_draws(), 512, "every draw from an all-spread buffer must tick spread_draws");
    assert_eq!(buf.compact_draws(), 0, "…and never compact_draws");
}

#[test]
fn augment_false_ticks_neither_counter() {
    // R266/F-P1/N1 — the disarmed-lever direction (the b349ec4/R249 precedent, ported to
    // this gate): `augment=false` never consults `compact` at all (`sym_idx` is
    // unconditionally 0 in `sample.rs`), so nothing was exercised and nothing may be
    // counted. MUTATION THAT REDS IT: hoist the tick above the `if augment` branch.
    let mut buf = ReplayBuffer::new(1, ENC);
    buf.rng = StdRng::seed_from_u64(0xA5A5_0000);
    let n_cells = buf.sym_tables.n_cells;
    let row = discriminating_row(0..n_cells); // spread — would tick spread_draws if counted
    buf.push_impl(row.push_config(0.0)).unwrap();

    for _ in 0..64 {
        buf.sample_batch_core(1, false).unwrap();
    }
    assert_eq!(buf.compact_draws(), 0, "an unaugmented draw must not tick compact_draws");
    assert_eq!(buf.spread_draws(), 0, "…nor spread_draws — the lever was never exercised");
}

#[test]
fn sample_batch_with_pos_core_ticks_the_same_counters() {
    // R266/F-P1/N1 — the counted helper is the ONE call site both sample cores route
    // through (`sample.rs::record_symmetry_draw`), so `sample_batch_with_pos_core` must
    // tick the SAME atomics as `sample_batch_core`, not a second, independent pair.
    // MUTATION THAT REDS IT: give the pos-variant its own uncounted `draw_record_sym` call.
    let mut buf = ReplayBuffer::new(1, ENC);
    buf.rng = StdRng::seed_from_u64(0x905E_C0DE);
    let n_cells = buf.sym_tables.n_cells;
    let dropped: HashSet<usize> =
        buf.sym_tables.dropped_cells.iter().map(|&c| c as usize).collect();
    let row = discriminating_row((0..n_cells).filter(|c| !dropped.contains(c)));
    buf.push_impl(row.push_config(0.0)).unwrap();
    assert_eq!(buf.compact_at(0), 1, "test setup: the row must be compact");

    for _ in 0..32 {
        buf.sample_batch_with_pos_core(1, true).unwrap();
    }
    assert_eq!(buf.compact_draws(), 32, "the pos-variant must tick the SHARED counter");
    assert_eq!(buf.spread_draws(), 0);
}

// ── operator pin 3: no clipped copy is ever emitted ────────────────────────────

/// Per-channel mass of one emitted record. A clipped copy loses mass in at least
/// one of these five; an exact transform conserves all five.
#[derive(Debug, PartialEq)]
struct Mass {
    policy_sum: f32,
    state_nonzero: usize,
    chain_nonzero: usize,
    own_non_neutral: usize,
    wl_nonzero: usize,
}

fn mass(
    state: &[u16],
    chain: &[u16],
    policy: &[f32],
    own: &[u8],
    wl: &[u8],
) -> Mass {
    Mass {
        // Small integers only (see `mixed_row`), so f32 addition is exact and
        // summation order cannot perturb the comparison.
        policy_sum: policy.iter().sum(),
        state_nonzero: state.iter().filter(|&&v| v != 0).count(),
        chain_nonzero: chain.iter().filter(|&&v| v != 0).count(),
        own_non_neutral: own.iter().filter(|&&v| v != 1).count(),
        wl_nonzero: wl.iter().filter(|&&v| v != 0).count(),
    }
}

/// A record with content on `cells`, values chosen so every mass above is a small
/// exact integer.
fn mixed_row(cells: &[usize]) -> Row {
    let (_, _, po, ax, n_cells) = shape();
    let mut row = Row::neutral();
    for &cell in cells {
        row.state[cell] = f16::from_f32(1.0);
        row.chain[cell] = f16::from_f32(1.0);
        row.policy[cell] = 1.0;
        row.own[cell % ax] = 2;
        row.wl[cell % ax] = 1;
    }
    if po > n_cells {
        row.policy[n_cells] = 1.0;
    }
    row
}

/// Build a MIXED buffer whose slot `i` carries outcome `i` (outcomes ride through
/// the sample path unscattered, so they identify the source slot of each emitted
/// record). Even slots are compact, odd slots are spread.
fn mixed_buffer(seed: u64) -> (ReplayBuffer, Vec<Mass>) {
    let (st, ch, po, ax, _) = shape();
    let mut buf = ReplayBuffer::new(8, ENC);
    buf.rng = StdRng::seed_from_u64(seed);
    let dropped: Vec<usize> =
        buf.sym_tables.dropped_cells.iter().map(|&c| c as usize).collect();
    let interior: Vec<usize> = {
        let d: HashSet<usize> = dropped.iter().copied().collect();
        (0..buf.sym_tables.n_cells).filter(|c| !d.contains(c)).take(5).collect()
    };

    let mut masses = Vec::with_capacity(8);
    for slot in 0..8 {
        let row = if slot % 2 == 0 {
            mixed_row(&interior)
        } else {
            let mut cells = interior.clone();
            cells.push(dropped[slot % dropped.len()]);
            mixed_row(&cells)
        };
        buf.push_impl(row.push_config(slot as f32)).unwrap();
        assert_eq!(
            buf.compact_at(slot),
            u8::from(slot % 2 == 0),
            "test setup: slot {slot} must be {}",
            if slot % 2 == 0 { "compact" } else { "spread" }
        );
        let s = slot * st;
        let c = slot * ch;
        let p = slot * po;
        let a = slot * ax;
        masses.push(mass(
            &buf.states[s..s + st],
            &buf.chain_planes[c..c + ch],
            &buf.policies[p..p + po],
            &buf.ownership[a..a + ax],
            &buf.winning_line[a..a + ax],
        ));
    }
    // Anti-vacuity: the buffer really is mixed.
    assert!(masses.len() == 8 && buf.compact_at(0) == 1 && buf.compact_at(1) == 0);
    (buf, masses)
}

#[test]
fn sample_batch_core_never_emits_a_clipped_copy() {
    let (st, ch, po, ax, _) = shape();
    let (mut buf, masses) = mixed_buffer(0xA110_5500);

    for _ in 0..40 {
        let out = buf.sample_batch_core(64, true).unwrap();
        for b in 0..out.batch_size {
            let src = out.outcomes[b] as usize;
            let got = mass(
                &out.states[b * st..(b + 1) * st],
                &out.chain[b * ch..(b + 1) * ch],
                &out.policies[b * po..(b + 1) * po],
                &out.ownership[b * ax..(b + 1) * ax],
                &out.winning_line[b * ax..(b + 1) * ax],
            );
            assert_eq!(
                got, masses[src],
                "emitted record lost mass vs source slot {src} — a clipped copy reached \
                 the training batch"
            );
        }
    }
}

#[test]
fn sample_batch_with_pos_core_never_emits_a_clipped_copy() {
    let (st, ch, po, ax, _) = shape();
    let (mut buf, masses) = mixed_buffer(0xB220_6600);

    for _ in 0..40 {
        let out = buf.sample_batch_with_pos_core(64, true).unwrap();
        for b in 0..out.batch_size {
            let src = out.outcomes[b] as usize;
            let got = mass(
                &out.states[b * st..(b + 1) * st],
                &out.chain[b * ch..(b + 1) * ch],
                &out.policies[b * po..(b + 1) * po],
                &out.ownership[b * ax..(b + 1) * ax],
                &out.winning_line[b * ax..(b + 1) * ax],
            );
            assert_eq!(
                got, masses[src],
                "emitted record lost mass vs source slot {src} — a clipped copy reached \
                 the training batch"
            );
        }
    }
}

/// Anti-vacuity for the two tests above: the mass metric DOES detect a clipped
/// copy. Applying a dropping element by hand to a spread record must lose mass —
/// if it did not, `never_emits_a_clipped_copy` would pass for free.
#[test]
fn the_mass_metric_detects_a_clipped_copy() {
    let (st, ch, po, ax, _) = shape();
    let (buf, masses) = mixed_buffer(0xC330_7700);
    let dropping = (0..N_SYMS)
        .find(|s| !WINDOW_PRESERVING_SYMS.contains(s))
        .expect("some element must drop");

    let slot = 1usize; // a spread slot
    let s = slot * st;
    let c = slot * ch;
    let p = slot * po;
    let a = slot * ax;

    let mut dst_state = vec![0u16; st];
    let mut dst_chain = vec![0u16; ch];
    let mut dst_pol = vec![0.0f32; po];
    let mut dst_own = vec![1u8; ax];
    let mut dst_wl = vec![0u8; ax];
    ReplayBuffer::apply_sym(
        dropping,
        ApplySymSlices {
            src: ApplySymSrc {
                state: &buf.states[s..s + st],
                chain: &buf.chain_planes[c..c + ch],
                policy: &buf.policies[p..p + po],
                own: &buf.ownership[a..a + ax],
                wl: &buf.winning_line[a..a + ax],
            },
            dst: ApplySymDst {
                state: &mut dst_state,
                chain: &mut dst_chain,
                policy: &mut dst_pol,
                own: &mut dst_own,
                wl: &mut dst_wl,
            },
            tables: buf.sym_tables,
        },
    );
    let clipped = mass(&dst_state, &dst_chain, &dst_pol, &dst_own, &dst_wl);
    assert_ne!(
        clipped, masses[slot],
        "sym {dropping} applied to a SPREAD record must lose mass — otherwise the \
         no-clipped-copy tests are vacuous"
    );
}
