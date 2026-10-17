//! R8-justify: the HEXG v4 field's contract (insert, load, round trip, sampled label) shares one row and the golden's offsets.

use std::collections::HashMap;

use mantis_selfplay::replay::hexg::{
    GraphRecord, HexgBuffer, FUTURE_EMPTY, FUTURE_HORIZON, FUTURE_MINE, FUTURE_THEIRS,
};
use mantis_selfplay::replay::sym::{rotate_axial, N_SYMS};

const ENC: &str = "gnn_axis_v1";
const VISIT_CAP: usize = 128;

/// Record 0 of the v4 golden: three stones at ply 3, its game ending at turn 7, eleven stones after it.
fn golden_row() -> GraphRecord {
    GraphRecord {
        stones: vec![(0, 0, 1), (1, 0, -1), (0, 1, 1)],
        visits: vec![(2, 0, 0.6), (0, 2, 0.4)],
        tail_mass: 0.0,
        current_player: 1,
        moves_remaining: 2,
        ply_index: 3,
        is_full_search: true,
        outcome: 1.0,
        value_valid: true,
        game_length: 7,
        game_id: -1,
        root_value: -0.625,
        root_value_valid: true,
        future_stones: Some(vec![
            (2, 0, 1),
            (0, 2, 1),
            (-1, 0, -1),
            (1, 1, -1),
            (2, -1, 1),
            (-1, 2, 1),
            (3, 0, -1),
            (0, 3, -1),
            (-2, 1, 1),
            (1, -2, 1),
            (2, 2, -1),
        ]),
        proven_tail: false,
    }
}

/// A row whose game runs well past the horizon, carrying its full twenty.
fn full_horizon_row() -> GraphRecord {
    let future: Vec<(i16, i16, i8)> = (0..FUTURE_HORIZON as i16)
        .map(|i| (i - 10, 3 + i % 2, if (i / 2) % 2 == 0 { 1 } else { -1 }))
        .collect();
    GraphRecord {
        ply_index: 3,
        game_length: 40,
        future_stones: Some(future),
        ..golden_row()
    }
}

fn unique_path(stem: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "hexg_future_{stem}_{}_{nanos}.hexg",
        std::process::id()
    ))
}

fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/replay")
        .join(name)
}

/// PLANTED BREAK — drop any arm of `validate_future_stones` and its case is inserted instead of refused.
#[test]
fn a_future_field_that_is_not_its_games_next_placements_is_refused_at_insert() {
    let with = |future: Vec<(i16, i16, i8)>, game_length: u16| GraphRecord {
        future_stones: Some(future),
        game_length,
        ..golden_row()
    };
    let eleven = golden_row().future_stones.unwrap();
    let mut twenty_one = full_horizon_row().future_stones.unwrap();
    twenty_one.push((9, 9, 1));
    let mut on_a_stone = eleven.clone();
    on_a_stone[4] = (1, 0, 1);
    let mut twice = eleven.clone();
    twice[5] = (2, 0, -1);
    let mut no_player = eleven.clone();
    no_player[0] = (2, 0, 0);
    let mut out_of_turn = eleven.clone();
    out_of_turn[1] = (0, 2, -1);
    let mut buf = HexgBuffer::new(4, ENC, VISIT_CAP).unwrap();
    for (case, rec) in [
        ("over the horizon", with(twenty_one, 40)),
        ("on a stone", with(on_a_stone, 7)),
        ("a cell twice", with(twice, 7)),
        ("no player", with(no_player, 7)),
        ("out of turn", with(out_of_turn, 7)),
        (
            "the mover's turn read as the opponent's",
            GraphRecord {
                current_player: -1,
                ..golden_row()
            },
        ),
        ("short of a longer game", with(eleven.clone(), 8)),
        ("past a shorter game", with(eleven.clone(), 6)),
        (
            "a full horizon past its game",
            GraphRecord {
                game_length: 11,
                ..full_horizon_row()
            },
        ),
    ] {
        let err = buf
            .push_record_impl(&rec, -1)
            .expect_err("a field that is not the game's next placements must be refused");
        assert!(err.contains("future-occupancy"), "{case}: {err}");
    }
    assert_eq!(buf.size(), 0, "no refused row may have been inserted");
    // The controls: absent, ending exactly where the game does, and the full horizon of a longer game.
    for rec in [
        GraphRecord {
            future_stones: None,
            ..golden_row()
        },
        golden_row(),
        with(eleven[..10].to_vec(), 7),
        full_horizon_row(),
        GraphRecord {
            game_length: 12,
            ..full_horizon_row()
        },
    ] {
        buf.push_record_impl(&rec, -1)
            .expect("the game's own next placements are admissible");
    }
    assert_eq!(buf.record_at(1), golden_row(), "record_at inverts the push");
}

/// Offsets into the v4 golden's record 0: after the 47-byte header, its 35-byte v3 head, three stones and two visits.
const R0: usize = 47;
const FLAG: usize = R0 + 35;
const COUNT: usize = FLAG + 1;
const FUTURE0: usize = COUNT + 1 + 3 * 5 + 2 * 8;
const GAME_LENGTH: usize = R0 + 14;

/// PLANTED BREAK — a v4 file whose field breaks the contract is refused at load and the buffer is untouched.
#[test]
fn a_v4_future_field_outside_its_contract_is_refused_at_load() {
    let golden = std::fs::read(fixture_path("hexg_v4_golden.hexg")).unwrap();
    assert_eq!((golden[FLAG], golden[COUNT]), (1, 11));
    assert_eq!(&golden[FUTURE0..FUTURE0 + 5], &[2, 0, 0, 0, 1]);
    let patched = |edits: &[(usize, &[u8])]| {
        let mut bytes = golden.clone();
        for &(at, with) in edits {
            bytes[at..at + with.len()].copy_from_slice(with);
        }
        bytes
    };
    for (stem, bytes) in [
        ("flag_two", patched(&[(FLAG, &[2])])),
        ("absent_with_stones", patched(&[(FLAG, &[0])])),
        ("over_the_horizon", patched(&[(COUNT, &[21])])),
        ("no_player", patched(&[(FUTURE0 + 4, &[0])])),
        ("out_of_turn", patched(&[(FUTURE0 + 9, &[0xFF])])),
        ("on_a_stone", patched(&[(FUTURE0, &[1, 0, 0, 0])])),
        ("a_cell_twice", patched(&[(FUTURE0 + 5, &[2, 0, 0, 0])])),
        (
            "game_length",
            patched(&[(GAME_LENGTH, &8u16.to_le_bytes())]),
        ),
    ] {
        let path = unique_path(stem);
        std::fs::write(&path, &bytes).unwrap();
        let mut buf = HexgBuffer::new(16, ENC, VISIT_CAP).unwrap();
        buf.push_record_impl(&full_horizon_row(), 7)
            .expect("seed push");
        let before = buf.record_at(0);
        let err = buf
            .load_from_path_impl(path.to_str().unwrap())
            .expect_err("an out-of-contract future field must be refused at load");
        assert!(err.contains("HEXG load"), "{stem}: {err}");
        assert!(err.contains("future"), "{stem}: {err}");
        assert_eq!(
            buf.record_at(0),
            before,
            "{stem}: a refused load must not touch a slot"
        );
        assert_eq!(
            buf.size(),
            1,
            "{stem}: a refused load must not change the ring"
        );
        let _ = std::fs::remove_file(path);
    }
}

/// Present (short and full) and absent rows survive a save and load field for field.
#[test]
fn the_field_round_trips_present_short_full_and_absent() {
    let rows = [
        golden_row(),
        full_horizon_row(),
        GraphRecord {
            future_stones: None,
            ..golden_row()
        },
        GraphRecord {
            future_stones: Some(vec![]),
            ply_index: 14,
            game_length: 7,
            ..golden_row()
        },
    ];
    let mut buf = HexgBuffer::new(8, ENC, VISIT_CAP).unwrap();
    for (i, rec) in rows.iter().enumerate() {
        buf.push_record_impl(rec, i as i64).unwrap();
    }
    let path = unique_path("roundtrip");
    buf.save_to_path_impl(path.to_str().unwrap()).unwrap();
    let mut back = HexgBuffer::new(8, ENC, VISIT_CAP).unwrap();
    assert_eq!(back.load_from_path_impl(path.to_str().unwrap()).unwrap(), 4);
    for (i, rec) in rows.iter().enumerate() {
        let want = GraphRecord {
            game_id: i as i64,
            ..rec.clone()
        };
        assert_eq!(back.record_at(i), want, "row {i}");
    }
    let _ = std::fs::remove_file(path);
}

/// Each legal node's label in one sampled graph, keyed by its coordinate.
fn labels_by_cell(g: &mantis_graph::AxisGraph, target: &[u8]) -> HashMap<(i32, i32), u8> {
    assert_eq!(target.len(), g.legal_node_gather.len());
    g.legal_node_gather
        .iter()
        .zip(target)
        .map(|(&row, &label)| {
            let row = row as usize;
            ((g.node_coords[row * 2], g.node_coords[row * 2 + 1]), label)
        })
        .collect()
}

/// Each legal cell's label names whose stone fills it, in the mover's frame; a row without the field labels none.
#[test]
fn the_sampled_label_names_whose_stone_fills_each_legal_cell() {
    let rows = [
        golden_row(),
        GraphRecord {
            current_player: -1,
            future_stones: golden_row()
                .future_stones
                .map(|f| f.into_iter().map(|(q, r, p)| (q, r, -p)).collect()),
            ..golden_row()
        },
        GraphRecord {
            future_stones: None,
            ..golden_row()
        },
    ];
    for (k, rec) in rows.iter().enumerate() {
        let mut buf = HexgBuffer::new(4, ENC, VISIT_CAP).unwrap();
        buf.push_record_impl(rec, 0).unwrap();
        let (graphs, t) = buf.sample_graph_batch_impl(1, false, 1).unwrap();
        let labels = labels_by_cell(&graphs[0], &t.future_target);
        let future: HashMap<(i32, i32), i8> = rec
            .future_stones
            .iter()
            .flatten()
            .map(|&(q, r, p)| ((i32::from(q), i32::from(r)), p))
            .collect();
        assert_eq!(t.future_valid, vec![u8::from(rec.future_stones.is_some())]);
        for (cell, &label) in &labels {
            let want = match future.get(cell) {
                None => FUTURE_EMPTY,
                Some(&p) if p == rec.current_player => FUTURE_MINE,
                Some(_) => FUTURE_THEIRS,
            };
            assert_eq!(label, want, "row {k} cell {cell:?}");
        }
        assert!(
            future.keys().all(|c| labels.contains_key(c)),
            "row {k}: every future stone of the fixture lies in the row's legal set"
        );
    }
}

/// PLANTED BREAK — a sampler that rotates the stones but not the field: some D6 element must map the labels across.
#[test]
fn the_label_follows_its_cell_under_every_d6_element() {
    let rec = golden_row();
    let mut buf = HexgBuffer::new(4, ENC, VISIT_CAP).unwrap();
    buf.push_record_impl(&rec, 0).unwrap();
    let (base_graphs, base_t) = buf.sample_graph_batch_impl(1, false, 1).unwrap();
    let base = labels_by_cell(&base_graphs[0], &base_t.future_target);
    let mut elements_seen = std::collections::HashSet::new();
    buf.seed_sampler(5);
    for _ in 0..96 {
        let (graphs, t) = buf.sample_graph_batch_impl(1, true, 1).unwrap();
        let got = labels_by_cell(&graphs[0], &t.future_target);
        let explained: Vec<usize> = (0..N_SYMS)
            .filter(|&s| {
                base.iter().all(|(&(q, r), &label)| {
                    got.get(&rotate_axial(q, r, s)).copied() == Some(label)
                }) && got.len() == base.len()
            })
            .collect();
        assert!(
            !explained.is_empty(),
            "no D6 element maps the row's labels onto the sampled ones"
        );
        elements_seen.extend(explained);
    }
    assert_eq!(
        elements_seen.len(),
        N_SYMS,
        "the draw must reach every element"
    );
}
