"""Every dump keeps its own artifact, and the rate bar still reads its time off the name."""
from __future__ import annotations

import time
from pathlib import Path

from mantis.diagnostics.f816_37_rate_bar import scan_dumps
from mantis.selfplay.collate_dump import write_collate_dump
from mantis.selfplay.graph_collate import GraphContractError


class _Wire:
    edge_offsets = [0, 3]
    node_offsets = [0, 2]


def test_two_dumps_of_one_round_in_one_millisecond_are_both_kept(tmp_path: Path, monkeypatch) -> None:
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now)
    paths = [write_collate_dump(_Wire(), dump_dir=tmp_path, context={"round_id": "r1"}, error=GraphContractError("x"))
             for _ in range(2)]
    assert None not in paths and len(set(paths)) == 2
    assert len(list(tmp_path.glob("collate_dump_*.json"))) == 2
    assert all(abs(f.when - int(now * 1000) / 1000.0) < 1e-6 for f in scan_dumps(tmp_path))
