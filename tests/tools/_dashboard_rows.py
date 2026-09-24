"""The dashboard tests' shared event-row writers: a JSONL events file and the ladder-state document."""
from __future__ import annotations

import json
from pathlib import Path


def write_events(tmp_path: Path, rows: list[dict], name: str = "events.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def ladder_state(history: list[tuple[int, int, float]]) -> dict:
    return {"sealbot_d5": {"name": "sealbot_d5", "status": "active", "consec": 0,
                           "history": [{"round_idx": i, "games": g, "wr": wr, "ci_lo": None}
                                       for i, g, wr in history]}}
