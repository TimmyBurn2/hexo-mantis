//! The recorded radius-8 leaf positions: an `S` line per position (to move, stones to place, count, `q r player` triples), then an `M` line of its game's moves, unread here.

use std::fmt::Write as _;
use std::path::PathBuf;

use sha2::{Digest, Sha256};

/// The fixture's sha256, so an edit cannot silently move the corpus the parity test and the bench read.
pub const POSITIONS_R8_SHA256: &str =
    "d5c5a7ba032c82f3f71309f2b676c81d3ad076fbf4af374c3978c4d791c6c1c7";

/// The number of positions the fixture holds; fewer is a shrunken corpus, never a pass.
pub const POSITIONS_R8_COUNT: usize = 512;

/// One recorded position: the side to move, the stones it still places this turn, the stones on the board.
pub struct RecordedPosition {
    pub to_move: i8,
    pub moves_remaining: u8,
    pub stones: Vec<(i32, i32, i8)>,
}

/// The fixture's path.
#[must_use]
pub fn positions_r8_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/graph_build/positions_r8.txt")
}

/// Every position, or an error naming the absent file, the sha drift, the count or the first bad line.
pub fn read_positions_r8() -> Result<Vec<RecordedPosition>, String> {
    let path = positions_r8_path();
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut sha = String::with_capacity(64);
    for b in Sha256::digest(&bytes) {
        write!(sha, "{b:02x}").map_err(|e| e.to_string())?;
    }
    if sha != POSITIONS_R8_SHA256 {
        return Err(format!(
            "{}: sha256 {sha}, pinned {POSITIONS_R8_SHA256}",
            path.display()
        ));
    }
    let text = String::from_utf8(bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::with_capacity(POSITIONS_R8_COUNT);
    for (no, line) in text.lines().enumerate() {
        let bad = |what: &str| format!("{} line {}: {what}", path.display(), no + 1);
        let mut fields = line.split_ascii_whitespace();
        match fields.next() {
            Some("M") => continue,
            Some("S") => {}
            _ => return Err(bad("neither an S nor an M record")),
        }
        let v: Vec<i32> = fields
            .map(str::parse)
            .collect::<Result<_, _>>()
            .map_err(|e| bad(&format!("{e}")))?;
        let [to_move, moves_remaining, n, ref rest @ ..] = v[..] else {
            return Err(bad(
                "an S record needs to-move, moves-remaining and a count",
            ));
        };
        if usize::try_from(n).ok().and_then(|n| n.checked_mul(3)) != Some(rest.len()) {
            return Err(bad("the stone count disagrees with the triples"));
        }
        let mut stones = Vec::with_capacity(rest.len() / 3);
        for t in rest.chunks_exact(3) {
            let player = i8::try_from(t[2]).map_err(|e| bad(&format!("{e}")))?;
            stones.push((t[0], t[1], player));
        }
        out.push(RecordedPosition {
            to_move: i8::try_from(to_move).map_err(|e| bad(&format!("{e}")))?,
            moves_remaining: u8::try_from(moves_remaining).map_err(|e| bad(&format!("{e}")))?,
            stones,
        });
    }
    if out.len() != POSITIONS_R8_COUNT {
        return Err(format!(
            "{}: {} positions, want {POSITIONS_R8_COUNT}",
            path.display(),
            out.len()
        ));
    }
    Ok(out)
}
