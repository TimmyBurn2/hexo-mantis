//! INV15 — v6w25 / v6 encode round-trip regression pins (ported from the frozen
//! engine's inv15_v6w25_encode_roundtrip.rs; the 3 pins are byte-verbatim,
//! rewired to the mantis-encoding FREE-FN kernels + mantis-core Board + lookup).
//!
//!   1. v6w25 corner-stone byte identity through `encode_state_to_buffer_channels`.
//!   2. v6 byte identity through `encode_state_to_buffer_channels`.
//!   3. v6w25 chain-plane axis-run math through `encode_chain_planes`.
//!
//! Tests exercise the kernels directly; they do NOT route through `to_planes` /
//! `to_planes_channels`, which are α-deferred (panic) for multi-window encodings.

use mantis_core::board::{BOARD_SIZE, HALF, TOTAL_CELLS};
use mantis_core::Board;
use mantis_encoding::{encode_chain_planes, encode_state_to_buffer_channels, lookup_or_panic};

/// v6w25 corner-stone byte identity through `encode_state_to_buffer_channels`.
#[test]
fn test_v6w25_encode_state_corner_stone_byte_identity() {
    let spec = lookup_or_panic("v6w25");
    let n_cells = spec.n_cells(); // 625 = 25 * 25
    let kept = spec.kept_plane_indices; // &[0,1,2,3,8,9,10,11]
    assert_eq!(n_cells, 625, "v6w25 n_cells precondition");
    assert_eq!(kept.len(), 8, "v6w25 kept_plane_indices len precondition");

    let mut planes_2 = vec![0.0f32; 2 * n_cells];
    planes_2[624] = 1.0; // my-stone plane: corner
    planes_2[n_cells + 623] = 1.0; // opp-stone plane: adjacent corner

    let board = Board::new();
    let mut out = vec![0.0f32; kept.len() * n_cells]; // 8 * 625 = 5000
    encode_state_to_buffer_channels(&board, &planes_2, &mut out, kept, n_cells);

    assert_eq!(out[624], 1.0, "v6w25: my-stone corner cell must survive at flat 624 (slot 0)");
    assert_eq!(out[4 * n_cells + 623], 1.0, "v6w25: opp-stone cell must land at flat 623 of slot 4");
    assert_eq!(out[0], 0.0, "v6w25: my-plane flat 0 must be untouched");
    assert_eq!(out[4 * n_cells + 624], 0.0, "v6w25: opp-plane flat 624 must be untouched");
    for v in &out[n_cells..2 * n_cells] {
        assert_eq!(*v, 0.0, "history plane slot 1 must be zeroed");
    }

    let bcast_channels = [0usize, 8, 16, 17];
    let mut bcast_out = vec![0.0f32; bcast_channels.len() * n_cells];
    encode_state_to_buffer_channels(&board, &planes_2, &mut bcast_out, &bcast_channels, n_cells);
    assert_eq!(
        bcast_out[2 * n_cells],
        bcast_out[2 * n_cells + 624],
        "v6w25: moves_remaining broadcast (ch 16) must be plane-uniform"
    );
    assert_eq!(
        bcast_out[3 * n_cells],
        bcast_out[3 * n_cells + 624],
        "v6w25: ply parity broadcast (ch 17) must be plane-uniform"
    );
}

/// v6 byte identity through `encode_state_to_buffer_channels`.
#[test]
fn test_v6_encode_state_byte_identity_unchanged() {
    let mut board = Board::new();
    board.apply_move(0, 0).unwrap(); // P1 places at origin

    let (views, _centers) = board.get_cluster_views();
    assert!(!views.is_empty(), "v6 board with one stone must yield one cluster view");
    let planes_2 = &views[0];
    assert_eq!(planes_2.len(), 2 * TOTAL_CELLS, "v6 cluster view must be 2 * 361 = 722 floats");

    // After P1's move, current_player is Player::Two; the P1 stone at origin
    // lives in the OPPONENT plane. flat_idx for v6 = 9*19+9 = 180.
    let origin_flat = (HALF as usize) * BOARD_SIZE + (HALF as usize);
    assert_eq!(origin_flat, 180, "v6 origin flat-idx precondition");
    assert_eq!(planes_2[origin_flat], 0.0, "v6 current-player (P2) plane: origin must be empty");
    assert_eq!(planes_2[TOTAL_CELLS + origin_flat], 1.0, "v6 opponent plane: P1 stone at origin");

    let v6_spec = lookup_or_panic("v6");
    let kept = v6_spec.kept_plane_indices;
    assert_eq!(v6_spec.n_cells(), TOTAL_CELLS, "v6 n_cells = 361");

    let mut out = vec![0.0f32; kept.len() * TOTAL_CELLS];
    encode_state_to_buffer_channels(&board, planes_2, &mut out, kept, TOTAL_CELLS);

    let ch8_slot = kept
        .iter()
        .position(|&c| c == 8)
        .expect("v6 kept_plane_indices must contain channel 8");
    assert_eq!(
        out[ch8_slot * TOTAL_CELLS + origin_flat],
        1.0,
        "v6 byte identity: opp-stone (P1) at origin survives slot lookup"
    );
    let ch0_slot = kept
        .iter()
        .position(|&c| c == 0)
        .expect("v6 kept_plane_indices must contain channel 0");
    assert_eq!(
        out[ch0_slot * TOTAL_CELLS + origin_flat],
        0.0,
        "v6 byte identity: my-stone (P2) plane is empty at origin"
    );

    let bcast_channels = [0usize, 8, 16, 17];
    let mut bcast_out = vec![0.0f32; bcast_channels.len() * TOTAL_CELLS];
    encode_state_to_buffer_channels(&board, planes_2, &mut bcast_out, &bcast_channels, TOTAL_CELLS);
    assert_eq!(
        bcast_out[2 * TOTAL_CELLS],
        bcast_out[2 * TOTAL_CELLS + (TOTAL_CELLS - 1)],
        "v6: moves_remaining broadcast (ch 16) must be plane-uniform"
    );
    assert_eq!(
        bcast_out[3 * TOTAL_CELLS],
        bcast_out[3 * TOTAL_CELLS + (TOTAL_CELLS - 1)],
        "v6: ply parity broadcast (ch 17) must be plane-uniform"
    );
}

/// v6w25 chain-plane axis-run math through `encode_chain_planes`.
#[test]
fn test_v6w25_encode_chain_planes_axis_runs() {
    let spec = lookup_or_panic("v6w25");
    let n_cells = spec.n_cells(); // 625
    let trunk_sz = spec.trunk_size as i32; // 25
    assert_eq!(n_cells, 625, "v6w25 n_cells precondition");
    assert_eq!(trunk_sz, 25, "v6w25 trunk_size precondition");
    let half: i32 = (trunk_sz - 1) / 2; // 12

    let flat = |q: i32, r: i32| -> usize {
        ((q + half) as usize) * (trunk_sz as usize) + ((r + half) as usize)
    };

    let mut cur = vec![0.0f32; n_cells];
    let mut opp = vec![0.0f32; n_cells];
    for q in [0i32, 1, 2] {
        cur[flat(q, 0)] = 1.0;
    }
    opp[flat(1, 1)] = 1.0;

    let mut out = vec![0.0f32; 6 * n_cells];
    encode_chain_planes(&cur, &opp, &mut out, n_cells, trunk_sz);

    // HEX_AXES: 0 = E(1,0), 1 = NE(0,1), 2 = (1,-1). For axis i: cur plane at
    // [2i*n_cells..], opp plane at [(2i+1)*n_cells..].
    let a0_cur = &out[0..n_cells];
    let a1_cur = &out[2 * n_cells..3 * n_cells];
    let a1_opp = &out[3 * n_cells..4 * n_cells];

    let three_sixths = 3.0 / 6.0;
    let one_sixth = 1.0 / 6.0;
    for q in [0i32, 1, 2] {
        let v = a0_cur[flat(q, 0)];
        assert!((v - three_sixths).abs() < 1e-5, "v6w25 E-axis cell ({q},0)={v} (expected 3/6)");
    }
    let v00_ne = a1_cur[flat(0, 0)];
    assert!((v00_ne - one_sixth).abs() < 1e-5, "v6w25 NE solo (0,0)={v00_ne} (expected 1/6)");
    let v11_ne_opp = a1_opp[flat(1, 1)];
    assert!((v11_ne_opp - one_sixth).abs() < 1e-5, "v6w25 NE opp (1,1)={v11_ne_opp} (expected 1/6)");

    for axis_i in 0..3 {
        let cur_plane = &out[(2 * axis_i) * n_cells..(2 * axis_i + 1) * n_cells];
        let opp_plane = &out[(2 * axis_i + 1) * n_cells..(2 * axis_i + 2) * n_cells];
        assert_eq!(cur_plane[624], 0.0, "axis {axis_i}: corner cur cell must be zero");
        assert_eq!(opp_plane[624], 0.0, "axis {axis_i}: corner opp cell must be zero");
    }
}
