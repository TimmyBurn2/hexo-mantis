"""Pure-numpy reader for a HEXG v2 replay ring; the layout is `replay/hexg/persist.rs`'s."""
from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEXG_MAGIC = 0x48455847
HEXG_VERSION = 2
# Header (LE): magic u32, version u32, max_stones u32, max_visits u32, capacity u64, size u64,
# name_len u32, name bytes; then `size` records oldest first, each a FIXED head + stones + visits.
FIXED = struct.Struct("<HHbBHBBfHqHf")
# Record head: n_stones u16, n_visits u16, current_player i8, moves_remaining u8, ply_index u16,
# is_full_search u8, value_valid u8, outcome f32, game_length u16, game_id i64, weight u16, tail f32.
STONE_DT = np.dtype([("q", "<i2"), ("r", "<i2"), ("p", "i1")])
#: A visit entry is an improved-policy MASS on an explicit root child: the row stores no counts, no Q.
VISIT_DT = np.dtype([("q", "<i2"), ("r", "<i2"), ("prob", "<f4")])


@dataclass
class RingHeader:
    """The HEXG v2 header fields."""

    max_stones: int
    max_visits: int
    capacity: int
    size: int
    encoding: str


@dataclass
class Ring:
    """Column-major view of every record; `stone_off`/`visit_off` index the flat arrays."""

    header: RingHeader
    current_player: np.ndarray
    moves_remaining: np.ndarray
    ply_index: np.ndarray
    is_full_search: np.ndarray
    value_valid: np.ndarray
    outcome: np.ndarray
    game_length: np.ndarray
    game_id: np.ndarray
    weight_bits: np.ndarray
    tail_mass: np.ndarray
    n_stones: np.ndarray
    n_visits: np.ndarray
    stone_off: np.ndarray
    visit_off: np.ndarray
    stones: np.ndarray
    visits: np.ndarray

    def row_stones(self, i: int) -> np.ndarray:
        """Row `i`'s stones as a structured (q, r, p) array."""
        return self.stones[self.stone_off[i] : self.stone_off[i] + self.n_stones[i]]

    def row_visits(self, i: int) -> np.ndarray:
        """Row `i`'s visit entries as a structured (q, r, prob) array."""
        return self.visits[self.visit_off[i] : self.visit_off[i] + self.n_visits[i]]


def _read_header(buf: memoryview) -> tuple[RingHeader, int]:
    """Parse the header; returns it with the byte offset of the first record."""
    magic, version, max_stones, max_visits = struct.unpack_from("<IIII", buf, 0)
    if magic != HEXG_MAGIC:
        raise ValueError(f"bad magic {magic:#x}")
    if version != HEXG_VERSION:
        raise ValueError(f"HEXG version {version}, reader is v{HEXG_VERSION}")
    capacity, size, name_len = struct.unpack_from("<QQI", buf, 16)
    name = bytes(buf[36 : 36 + name_len]).decode("utf-8")
    return RingHeader(max_stones, max_visits, capacity, size, name), 36 + name_len


def load_ring(path: Path) -> Ring:
    """Parse every record of the ring at `path` into column arrays.

    Raises:
        ValueError: a wrong magic or version, a record over the header's caps, a truncated payload.
        OSError: the file cannot be read.
    """
    data = path.read_bytes()
    buf = memoryview(data)
    header, pos = _read_header(buf)
    n = header.size
    fixed = np.zeros((n, 12), dtype=np.float64)
    fixed_i = np.zeros((n, 12), dtype=np.int64)
    stone_chunks: list[np.ndarray] = []
    visit_chunks: list[np.ndarray] = []
    stone_off = np.zeros(n, dtype=np.int64)
    visit_off = np.zeros(n, dtype=np.int64)
    so = vo = 0
    for i in range(n):
        rec = FIXED.unpack_from(buf, pos)
        pos += FIXED.size
        ns, nv = rec[0], rec[1]
        if ns > header.max_stones or nv > header.max_visits:
            raise ValueError(f"record {i} declares {ns} stones / {nv} visits over cap")
        fixed_i[i, :] = [ns, nv, rec[2], rec[3], rec[4], rec[5], rec[6], 0, rec[8], rec[9], rec[10], 0]
        fixed[i, 7] = rec[7]
        fixed[i, 11] = rec[11]
        stone_chunks.append(np.frombuffer(buf, dtype=STONE_DT, count=ns, offset=pos))
        pos += ns * STONE_DT.itemsize
        visit_chunks.append(np.frombuffer(buf, dtype=VISIT_DT, count=nv, offset=pos))
        pos += nv * VISIT_DT.itemsize
        stone_off[i] = so
        visit_off[i] = vo
        so += ns
        vo += nv
    if pos != len(data):
        raise ValueError(f"trailing bytes: parsed to {pos}, file is {len(data)}")
    return Ring(
        header=header,
        n_stones=fixed_i[:, 0],
        n_visits=fixed_i[:, 1],
        current_player=fixed_i[:, 2],
        moves_remaining=fixed_i[:, 3],
        ply_index=fixed_i[:, 4],
        is_full_search=fixed_i[:, 5],
        value_valid=fixed_i[:, 6],
        outcome=fixed[:, 7].astype(np.float32),
        game_length=fixed_i[:, 8],
        game_id=fixed_i[:, 9],
        weight_bits=fixed_i[:, 10],
        tail_mass=fixed[:, 11].astype(np.float32),
        stone_off=stone_off,
        visit_off=visit_off,
        stones=np.concatenate(stone_chunks) if stone_chunks else np.zeros(0, STONE_DT),
        visits=np.concatenate(visit_chunks) if visit_chunks else np.zeros(0, VISIT_DT),
    )


def _hist(name: str, arr: np.ndarray) -> str:
    """One line: `name: value: count, …` over the distinct values of `arr`."""
    vals, counts = np.unique(arr, return_counts=True)
    body = ", ".join(f"{v}: {c}" for v, c in zip(vals.tolist(), counts.tolist(), strict=True))
    return f"{name}: {body}"


def main(argv: list[str]) -> int:
    """CLI: print the header and the per-field distributions of the ring at `argv[0]`."""
    ring = load_ring(Path(argv[0]))
    h = ring.header
    print(f"{argv[0]}: encoding={h.encoding} max_stones={h.max_stones} max_visits={h.max_visits} "
          f"capacity={h.capacity} size={h.size}")
    print(_hist("moves_remaining", ring.moves_remaining))
    print(_hist("is_full_search", ring.is_full_search))
    print(_hist("current_player", ring.current_player))
    print(_hist("value_valid", ring.value_valid))
    print(_hist("n_visits", ring.n_visits))
    print(f"ply_index: min {ring.ply_index.min()} med {np.median(ring.ply_index):.0f} max {ring.ply_index.max()}")
    print(f"n_stones: min {ring.n_stones.min()} med {np.median(ring.n_stones):.0f} max {ring.n_stones.max()}")
    print(f"tail_mass: mean {ring.tail_mass.mean():.5f} p50 {np.median(ring.tail_mass):.2e} "
          f"p90 {np.quantile(ring.tail_mass, 0.9):.2e} n(alpha==1) {(ring.tail_mass >= 1.0).sum()}")
    print(f"distinct game_id: {len(np.unique(ring.game_id))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
