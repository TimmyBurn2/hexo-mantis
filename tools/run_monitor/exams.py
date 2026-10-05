"""The exam positions read by a save's served net: each row's value is `sign · raw value` at its position, the mover's view."""
from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from typing import Any

from mantis._engine import Board
from mantis.util.loadpkg import load_tools_package

load_tools_package("dash", repo_root=Path(__file__).resolve().parents[2])
_engines = importlib.import_module("dash.engine.engines")


def load_positions(path: Path) -> list[dict[str, Any]]:
    """The exam rows `{"exam", "id", "stones", "sign"}`, in file order. Raises: ValueError (a malformed row), OSError."""
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not {"exam", "id", "stones", "sign"} <= set(row) or row["sign"] not in (1, -1):
            raise ValueError(f"{path}: line {i} is not an exam row: {line[:120]}")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no exam rows")
    return rows


def read_values(ckpt: Path, positions: list[dict[str, Any]], *, threads: int) -> list[dict[str, Any]]:
    """Each position's `sign · raw value` from the save's served net on CPU, the quiescence-off read. Raises: ValueError (a non-finite value), EngineLoadError, OSError."""
    eng = _engines.MantisEngine(_engines.EngineInfo(id=ckpt.stem, kind="mantis", path=str(ckpt)), device="cpu",
                                threads=threads)
    try:
        out = []
        for row in positions:
            board = Board.with_encoding_name(eng.encoding)
            for q, r in row["stones"]:
                board.apply_move(int(q), int(r))
            value = float(row["sign"]) * float(eng.raw_read(board).value)
            if not math.isfinite(value):
                raise ValueError(f"{ckpt.name}: exam {row['exam']} {row['id']} reads {value}")
            out.append({"exam": row["exam"], "id": row["id"], "value": value})
        return out
    finally:
        eng.close()


__all__ = ["load_positions", "read_values"]
