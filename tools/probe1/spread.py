"""Reading 6: the analyzer's symmetry spread (12 D6 maps about the first stone) over fixed positions × every checkpoint."""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from analyzer.engines import MANTIS, MantisEngine, discover
from analyzer.instruments import sweep


def spread_series(checkpoint_dir: Path, positions: list[dict[str, Any]], *, threads: int,
                  log: Callable[[str], None] = lambda _s: None) -> list[dict[str, Any]]:
    """Per stamped checkpoint: the orbit value spread per position and its distribution, the argmax agreement, the translation check."""
    out = []
    infos = [i for i in discover([checkpoint_dir]) if i.kind == MANTIS]
    for info in infos:
        t0 = time.perf_counter()
        engine = MantisEngine(info, device="cpu", threads=threads)
        try:
            rows = []
            for pos in positions:
                moves = [(int(q), int(r)) for q, r in pos["moves"]]
                s = sweep(engine, moves)
                rows.append({"row": pos["row"], "ply": pos["ply"], "spread": s["spread"], "value_min": s["value_min"],
                             "value_max": s["value_max"], "argmax_agreement": s["argmax_agreement"],
                             "translation_delta": (None if "absent" in s["translation"]
                                                   else round(abs(s["translation"]["value"] - s["rows"][0]["value"]), 4))})
        finally:
            engine.close()
        spreads = np.asarray([r["spread"] for r in rows], dtype=np.float64)
        agree = [int(r["argmax_agreement"].split("/")[0]) for r in rows]
        out.append({"engine": info.id, "step": info.step, "sha8": info.sha8, "n_positions": len(rows),
                    "spread_median": float(np.median(spreads)), "spread_mean": float(spreads.mean()),
                    "spread_max": float(spreads.max()), "spread_p90": float(np.quantile(spreads, 0.9)),
                    "argmax_agreement_mean": float(np.mean(agree)) / 12.0,
                    "argmax_full_agreement_share": float(np.mean([a == 12 for a in agree])),
                    "positions": rows, "ms": round((time.perf_counter() - t0) * 1000.0)})
        log(f"{info.id}: median spread {np.median(spreads):.4f} max {spreads.max():.4f} ({time.perf_counter() - t0:.0f} s)")
    return out


__all__ = ["spread_series"]
