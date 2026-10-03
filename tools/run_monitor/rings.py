"""The games a run produced after a save: the rows of a later ring whose game ids the save's ring had not reached."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from mantis._engine import HexgBuffer
from mantis.diagnostics.ring_reader import load_ring


def max_game_id(ring_path: Path) -> int:
    """The largest game id a ring holds; game ids only grow, so every later game's id is above it. Raises: ValueError, OSError."""
    ring = load_ring(ring_path)
    if len(ring.game_id) == 0:
        raise ValueError(f"{ring_path}: an empty ring has no last game")
    return int(ring.game_id.max())


def unseen_ring(later: Path, after_game_id: int, out: Path) -> dict[str, Any]:
    """Write the rows of `later` whose game id is above `after_game_id` as their own ring at `out`. Raises: ValueError (no such row), OSError."""
    ring = load_ring(later)
    keep = [i for i in range(len(ring.game_id)) if int(ring.game_id[i]) > after_game_id]
    if not keep:
        raise ValueError(f"{later}: no game after id {after_game_id}")
    buf = HexgBuffer(max(len(keep), 8), ring.header.encoding, ring.header.max_visits)
    for i in keep:
        buf.push_graph_position(
            [(int(s["q"]), int(s["r"]), int(s["p"])) for s in ring.row_stones(i)],
            [(int(v["q"]), int(v["r"]), float(v["prob"])) for v in ring.row_visits(i)],
            int(ring.current_player[i]), int(ring.moves_remaining[i]), int(ring.ply_index[i]),
            bool(ring.is_full_search[i]), float(ring.outcome[i]), bool(ring.value_valid[i]),
            int(ring.game_length[i]), game_id=int(ring.game_id[i]), tail_mass=float(ring.tail_mass[i]),
            root_value=float(ring.root_value[i]), root_value_valid=bool(ring.root_value_valid[i]))
    out.parent.mkdir(parents=True, exist_ok=True)
    buf.save_to_path(str(out))
    return {"source": str(later), "after_game_id": after_game_id, "rows": len(keep),
            "games": len({int(ring.game_id[i]) for i in keep})}


__all__ = ["max_game_id", "unseen_ring"]
