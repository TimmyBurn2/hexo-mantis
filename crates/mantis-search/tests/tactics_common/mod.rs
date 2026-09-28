//! The tactics suites' shared halves: the goldens fixture's rows, the boards they replay, the CHECK 1 port.

// Each suite reads a subset of these rows.
#![allow(dead_code)]

pub mod strict_check;

use mantis_core::Board;

/// Six's verdict at one budget, as TACTICS-DESIGN's T2 recorded it.
pub struct SixRow {
    pub found: bool,
    pub turns: u8,
    pub first: Option<((i32, i32), (i32, i32))>,
    pub nodes: u64,
    pub exhausted: bool,
}

/// One fixture position: its stratum, the moves that reach it and Six's verdict per `(turns, nodes)`.
pub struct GoldenRow {
    pub pid: String,
    pub set: String,
    pub class: String,
    pub cls: String,
    pub board: Board,
    pub six: Vec<((u8, u64), SixRow)>,
}

fn cell(v: &serde_json::Value) -> Option<(i32, i32)> {
    let a = v.as_array()?;
    Some((a.first()?.as_i64()? as i32, a.get(1)?.as_i64()? as i32))
}

/// Replay `stones` from an empty board at `radius` under the real cadence.
pub fn replay(stones: &[(i32, i32)], radius: i32) -> Board {
    let mut b = Board::new();
    b.set_legal_move_radius(radius);
    for &(q, r) in stones {
        b.apply_move(q, r)
            .expect("a recorded game's stone is placeable");
    }
    b
}

/// Every row of `tests/fixtures/tactics/turn_solver_goldens.jsonl`.
pub fn golden_rows() -> Vec<GoldenRow> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/tactics/turn_solver_goldens.jsonl"
    );
    let text = std::fs::read_to_string(path).expect("the goldens fixture is tracked");
    text.lines()
        .map(|line| {
            let v: serde_json::Value = serde_json::from_str(line).expect("a fixture row is JSON");
            let stones: Vec<(i32, i32)> = v["stones"]
                .as_array()
                .expect("stones")
                .iter()
                .map(|s| cell(s).expect("a stone is [q, r]"))
                .collect();
            let mut six: Vec<((u8, u64), SixRow)> = v["six"]
                .as_object()
                .expect("six")
                .iter()
                .map(|(k, s)| {
                    let (t, n) = k.split_once('/').expect("T/N");
                    let row = SixRow {
                        found: s["found"].as_i64() == Some(1),
                        turns: s["turns"].as_u64().expect("turns") as u8,
                        first: cell(&s["a"]).zip(cell(&s["b"])),
                        nodes: s["nodes"].as_u64().expect("nodes"),
                        exhausted: s["ex"].as_i64() == Some(1),
                    };
                    ((t.parse().expect("T"), n.parse().expect("N")), row)
                })
                .collect();
            six.sort_by_key(|(k, _)| *k);
            GoldenRow {
                pid: v["pid"].as_str().expect("pid").to_string(),
                set: v["set"].as_str().expect("set").to_string(),
                class: v["class"].as_str().expect("class").to_string(),
                cls: v["cls"].as_str().expect("cls").to_string(),
                board: replay(&stones, v["radius"].as_i64().expect("radius") as i32),
                six,
            }
        })
        .collect()
}
