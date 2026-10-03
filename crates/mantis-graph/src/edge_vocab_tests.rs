//! The vocabulary's size and order, its round trip through `edge_code`, and every built edge inside it.

use super::*;
use crate::{build_axis_graph, BuildParams, StoneList, EDGE_FEAT_DIM};

#[test]
fn the_vocabulary_is_the_dummy_then_every_axis_distance_and_player() {
    assert_eq!(edge_vocabulary(6).expect("w 6").len(), 91);
    assert_eq!(edge_vocabulary(5).expect("w 5").len(), 73);
    assert_eq!(
        edge_vocabulary(6).expect("w 6")[0].map(f32::to_bits),
        [0; EDGE_FEAT_DIM]
    );
}

#[test]
fn every_vocabulary_row_codes_to_its_own_index() {
    for w in [5, 6] {
        for (i, row) in edge_vocabulary(w).expect("vocab").iter().enumerate() {
            assert_eq!(
                edge_code(row, w).map(usize::from),
                Some(i),
                "w={w} row {row:?}"
            );
        }
    }
}

#[test]
fn a_row_outside_the_vocabulary_has_no_code() {
    let w = 6;
    for row in [
        [1.0, 0.0, 0.0, 2.5, 1.0], // fractional distance
        [1.0, 0.0, 0.0, 6.0, 1.0], // past the axis walk
        [1.0, 0.0, 0.0, 0.0, 1.0], // zero distance on a real edge
        [1.0, 1.0, 0.0, 2.0, 1.0], // two axes
        [0.0, 0.0, 0.0, 2.0, 1.0], // no axis
        [1.0, 0.0, 0.0, 2.0, 0.5], // not a player
        [0.0, 0.0, 0.0, 0.0, 1.0], // a dummy with a player
        [1.0, 0.0, 0.0, f32::NAN, 1.0],
    ] {
        assert_eq!(edge_code(&row, w), None, "{row:?}");
    }
    assert_eq!(edge_code(&[1.0, 0.0, 0.0, 2.0], w), None, "a short row");
}

#[test]
fn every_edge_a_real_build_emits_is_in_the_vocabulary_bit_for_bit() {
    let positions: [&[(i32, i32, i8)]; 3] = [
        &[(0, 0, 1), (1, 0, -1), (0, 1, -1)],
        &[
            (0, 0, 1),
            (2, -1, -1),
            (3, -1, 1),
            (5, 0, -1),
            (5, 1, 1),
            (4, 2, -1),
        ],
        &[],
    ];
    for w in [5u8, 6] {
        let vocab = edge_vocabulary(w).expect("vocab");
        for stones in positions {
            let g = build_axis_graph(
                &StoneList {
                    stones: stones.to_vec(),
                },
                &BuildParams {
                    win_length: w,
                    radius: 6,
                    current_player: 1,
                    moves_remaining: 2,
                    trunk_size: 19,
                },
            );
            for row in g.edge_attr.0.chunks(EDGE_FEAT_DIM) {
                let code =
                    edge_code(row, w).unwrap_or_else(|| panic!("w={w}: {row:?} has no code"));
                let back = vocab[usize::from(code)];
                assert_eq!(
                    back.map(f32::to_bits).to_vec(),
                    row.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }
}

#[test]
fn a_win_length_whose_vocabulary_overflows_a_byte_is_refused() {
    assert!(edge_vocabulary(16).is_none());
    assert_eq!(edge_vocabulary(15).map(|v| v.len()), Some(253));
    assert!(edge_vocabulary(1).is_none());
}
