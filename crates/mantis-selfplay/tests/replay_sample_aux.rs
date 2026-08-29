//! Dense sampling / aux oracle suite (relocated out of src, R5): weighted-sample
//! distribution (O-31), 12-sym aux equivariance (O-32), aux stress + reproject
//! (O-33), and the f16-bits sample(identity) preservation leg (O-34b). Ported
//! from the predecessor engine's `replay_buffer/{sample.rs,push.rs}` tests; the KILLed
//! v6 stride consts are replaced by `RegistrySpec` accessors.

use half::f16;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use mantis_core::Board;
use mantis_encoding::registry::lookup_or_panic;
use mantis_selfplay::replay::push_config::PushSingleConfig;
use mantis_selfplay::replay::sample::{ApplySymDst, ApplySymSlices, ApplySymSrc};
use mantis_selfplay::replay::sym::{SymTables, N_SYMS};
use mantis_selfplay::replay::ReplayBuffer;

/// v6 strides (state, chain, policy, aux, n_cells, n_planes) from the spec.
fn v6_shape() -> (usize, usize, usize, usize, usize, usize) {
    let s = lookup_or_panic("v6");
    (
        s.state_stride(),
        s.chain_stride(),
        s.policy_stride(),
        s.aux_stride(),
        s.n_cells(),
        s.n_planes,
    )
}

// ── O-31: weighted-sampling distribution (seeded) ────────────────────────────────

#[test]
fn o31_weighted_sampling_distribution() {
    let mut buf = ReplayBuffer::new(300, "v6");
    buf.rng = StdRng::seed_from_u64(42);
    // <10 → 0.15, <25 → 0.50, ≥25 → 1.0
    buf.set_weight_schedule(vec![10, 25], vec![0.15, 0.50], 1.0)
        .unwrap();

    for _ in 0..100 {
        buf.push_for_test(1.0, 5, true); // short → outcome 1.0
    }
    for _ in 0..100 {
        buf.push_for_test(2.0, 15, true); // medium → outcome 2.0
    }
    for _ in 0..100 {
        buf.push_for_test(3.0, 40, true); // long → outcome 3.0
    }
    assert_eq!(buf.size(), 300);

    let n_samples = 10_000;
    let (mut count_short, mut count_medium, mut count_long) = (0usize, 0usize, 0usize);
    for _ in 0..n_samples {
        let idx = buf.weighted_sample_one();
        match buf.outcomes[idx] as u32 {
            1 => count_short += 1,
            2 => count_medium += 1,
            3 => count_long += 1,
            _ => panic!("unexpected outcome"),
        }
    }

    let ratio_short_long = count_short as f64 / count_long as f64;
    let ratio_medium_long = count_medium as f64 / count_long as f64;
    assert!(
        ratio_short_long < 0.30,
        "short/long {ratio_short_long:.3} should be < 0.30 (~0.15)"
    );
    assert!(
        ratio_short_long > 0.05,
        "short/long {ratio_short_long:.3} should be > 0.05 (~0.15)"
    );
    assert!(
        ratio_medium_long < 0.80,
        "medium/long {ratio_medium_long:.3} should be < 0.80 (~0.50)"
    );
    assert!(
        ratio_medium_long > 0.25,
        "medium/long {ratio_medium_long:.3} should be > 0.25 (~0.50)"
    );
}

// ── O-32: aux augment equivariance across all 12 syms ────────────────────────────

#[test]
fn o32_aux_augment_equivariance() {
    let (_st, _ch, policy_stride, aux_stride, n_cells, n_planes) = v6_shape();
    let chain_planes = 6;
    let tables = SymTables::new();

    for &marker_src in &[0usize, 200, 180, 360] {
        for sym_idx in 0..N_SYMS {
            let mut src_state = vec![0u16; n_planes * n_cells];
            let src_chain = vec![0u16; chain_planes * n_cells];
            let mut src_pol = vec![0.0f32; policy_stride];
            let mut src_own = vec![1u8; aux_stride];
            let mut src_wl = vec![0u8; aux_stride];

            src_state[marker_src] = f16::from_f32(7.0).to_bits();
            src_pol[marker_src] = 7.0;
            src_own[marker_src] = 2; // P1
            src_wl[marker_src] = 1;

            let mut dst_state = vec![0u16; n_planes * n_cells];
            let mut dst_chain = vec![0u16; chain_planes * n_cells];
            let mut dst_pol = vec![0.0f32; policy_stride];
            let mut dst_own = vec![1u8; aux_stride];
            let mut dst_wl = vec![0u8; aux_stride];

            ReplayBuffer::apply_sym(
                sym_idx,
                ApplySymSlices {
                    src: ApplySymSrc {
                        state: &src_state,
                        chain: &src_chain,
                        policy: &src_pol,
                        own: &src_own,
                        wl: &src_wl,
                    },
                    dst: ApplySymDst {
                        state: &mut dst_state,
                        chain: &mut dst_chain,
                        policy: &mut dst_pol,
                        own: &mut dst_own,
                        wl: &mut dst_wl,
                    },
                    tables: &tables,
                },
            );

            let dst_state_idx = (0..n_cells).find(|&i| dst_state[i] != 0);
            let dst_pol_idx = (0..n_cells).find(|&i| dst_pol[i] != 0.0);
            let dst_own_idx = (0..aux_stride).find(|&i| dst_own[i] == 2);
            let dst_wl_idx = (0..aux_stride).find(|&i| dst_wl[i] == 1);

            assert_eq!(
                dst_state_idx, dst_pol_idx,
                "sym {sym_idx} src {marker_src}: state vs policy"
            );
            assert_eq!(
                dst_state_idx, dst_own_idx,
                "sym {sym_idx} src {marker_src}: state vs ownership"
            );
            assert_eq!(
                dst_state_idx, dst_wl_idx,
                "sym {sym_idx} src {marker_src}: state vs winning_line"
            );
        }
    }
}

// ── O-33: dense aux stress + reproject ───────────────────────────────────────────

#[test]
fn o33_aux_stress_1k_rows() {
    let (state_stride, chain_stride, policy_stride, aux_stride, _nc, _np) = v6_shape();
    let mut buf = ReplayBuffer::new(2000, "v6");
    for _ in 0..1000 {
        buf.push_for_test(0.0, 10, true);
    }
    assert_eq!(buf.size(), 1000);
    assert_eq!(buf.ownership.len(), 2000 * aux_stride);
    assert_eq!(buf.winning_line.len(), 2000 * aux_stride);

    let mut rng = StdRng::seed_from_u64(0xA1A1);
    for slot in 0..1000 {
        let a_start = slot * aux_stride;
        for _ in 0..20 {
            let cell = rng.random_range(0..aux_stride);
            buf.ownership[a_start + cell] = if rng.random_bool(0.5) { 0 } else { 2 };
            buf.winning_line[a_start + cell] = 1;
        }
    }

    let mut dst_state = vec![0u16; state_stride];
    let mut dst_chain = vec![0u16; chain_stride];
    let mut dst_pol = vec![0.0f32; policy_stride];
    let mut dst_own = vec![1u8; aux_stride];
    let mut dst_wl = vec![0u8; aux_stride];

    for _ in 0..1000 {
        let idx = buf.weighted_sample_one();
        let sym_idx = rng.random_range(0..N_SYMS);

        dst_state.fill(0);
        dst_chain.fill(0);
        dst_pol.fill(0.0);
        dst_own.fill(1);
        dst_wl.fill(0);

        let s = idx * state_stride;
        let c = idx * chain_stride;
        let p = idx * policy_stride;
        let a = idx * aux_stride;
        ReplayBuffer::apply_sym(
            sym_idx,
            ApplySymSlices {
                src: ApplySymSrc {
                    state: &buf.states[s..s + state_stride],
                    chain: &buf.chain_planes[c..c + chain_stride],
                    policy: &buf.policies[p..p + policy_stride],
                    own: &buf.ownership[a..a + aux_stride],
                    wl: &buf.winning_line[a..a + aux_stride],
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

        for &v in &dst_own {
            assert!(v <= 2, "ownership out-of-range: {v}");
        }
        for &v in &dst_wl {
            assert!(v <= 1, "winning_line out-of-range: {v}");
        }
    }
}

#[test]
fn o33_aux_alignment_reproject() {
    let (_st, _ch, _po, aux_stride, _nc, _np) = v6_shape();

    // 6-cell winning line at axial (5, 0)..(5, 5) — far from origin.
    let winning_cells: Vec<(i32, i32)> = (0..6).map(|i| (5, i)).collect();
    let (cq_a, cr_a) = (5i32, 2i32); // centre A (legacy bbox frame)
    let (cq_b, cr_b) = (8i32, 8i32); // centre B (per-cluster centre)

    let proj_a: Vec<usize> = winning_cells
        .iter()
        .map(|&(q, r)| Board::window_flat_idx_at(q, r, cq_a, cr_a))
        .collect();
    let proj_b: Vec<usize> = winning_cells
        .iter()
        .map(|&(q, r)| Board::window_flat_idx_at(q, r, cq_b, cr_b))
        .collect();

    let differs = proj_a
        .iter()
        .zip(&proj_b)
        .any(|(a, b)| *a < aux_stride && *b < aux_stride && a != b);
    assert!(
        differs,
        "test setup: centres A and B must yield different projections"
    );

    let mut buf = ReplayBuffer::new(4, "v6");
    let a_start = 0;
    for &flat in &proj_b {
        if flat < aux_stride {
            buf.winning_line[a_start + flat] = 1;
        }
    }
    buf.size = 1;
    buf.head = 1;

    let mut centre_b_hits = 0usize;
    for &flat in &proj_b {
        if flat < aux_stride {
            assert_eq!(
                buf.winning_line[a_start + flat],
                1,
                "centre-B projection cell {flat} must be marked"
            );
            centre_b_hits += 1;
        }
    }
    assert!(centre_b_hits > 0, "no centre-B cells landed in window");

    let mut centre_a_only_misses = 0usize;
    for (&fa, &fb) in proj_a.iter().zip(&proj_b) {
        if fa < aux_stride && fa != fb && buf.winning_line[a_start + fa] == 0 {
            centre_a_only_misses += 1;
        }
    }
    assert!(
        centre_a_only_misses > 0,
        "centre-A projection should diverge from centre-B on ≥1 cell"
    );
}

// ── O-34b: f16-bits preservation through sample(identity) ────────────────────────

#[test]
fn o34b_f16_bits_survive_sample_identity() {
    let (state_stride, chain_stride, policy_stride, aux_stride, _nc, _np) = v6_shape();
    let patterns: [u16; 4] = [0x7e00, 0x0001, 0x8000, 0x7bff]; // NaN, subnormal, -0, max-normal

    let mut state = vec![f16::from_f32(0.0); state_stride];
    let mut chain = vec![f16::from_f32(0.0); chain_stride];
    for (i, &bits) in patterns.iter().enumerate() {
        state[i] = f16::from_bits(bits);
        chain[i] = f16::from_bits(bits);
    }
    let policy = vec![0.0f32; policy_stride];
    let own = vec![1u8; aux_stride];
    let wl = vec![0u8; aux_stride];

    let mut buf = ReplayBuffer::new(4, "v6");
    buf.push_impl(PushSingleConfig {
        state: &state,
        chain_planes: &chain,
        policy: &policy,
        outcome: 0.0,
        ownership: &own,
        winning_line: &wl,
        game_id: -1,
        game_length: 0,
        is_full_search: true,
        position_index: 0,
        value_target_valid: true,
    })
    .unwrap();

    // augment=false → identity scatter → output bits == stored bits (no f16→f32→f16).
    let out = buf.sample_batch_core(1, false).unwrap();
    for (i, &bits) in patterns.iter().enumerate() {
        assert_eq!(
            out.states[i], bits,
            "sample(identity) state bit pattern {i}"
        );
        assert_eq!(out.chain[i], bits, "sample(identity) chain bit pattern {i}");
    }
}
