"""R339(c) — DUMP-ON-FIRE for the graph-contract check: save the evidence before the raise.

WHY THIS EXISTS. `F-816-37` fired ONCE, on the eval path, and killed a run: *"edge axis
one-hot is not a clean one-hot (edge 803217): [0.0, 0.0, 0.5]"*. Three diagnostics forcing the
check on every collate found nothing across ~1 000 collates, and a 90-minute burst produced
zero occurrences — so the class is rare, unreproduced and uncaused, and the ONE thing the next
occurrence must not do is take the run down while leaving nothing behind. R339(c) rules it an
INSTRUMENT rather than a hunt: the check runs 1-in-1 on the eval path, and when it fires the
offending batch is on disk before the exception propagates.

WHAT IS SAVED AND WHY EACH PIECE. The wire arrays are saved RAW — the same flat post-marshal
buffers the check read, not a reshaped view — because a reshape is a hypothesis about the
layout and the layout is one of the suspects. The graph ids and CSR offsets locate the
offending edge inside its own graph rather than in the fused batch, which is what makes a
per-graph re-derivation possible off-box. The CONTEXT (round, phase, concurrency in force) is
what R339(c) turns its HALT condition on: an occurrence under G > 1 and one under G = 1 are
different findings, and nothing else in the artifact records which one happened.

A DUMP MAY NEVER REPLACE THE RAISE. Every failure here is swallowed and reported in the
returned path's place — the original `GraphContractError` is what the caller re-raises. A
diagnostic that can convert a named contract failure into an `OSError` from the diagnostic
itself would destroy the very evidence it exists to keep (LAW-14 is about persistence being
fatal for RUN state; this is a post-mortem artifact and the opposite rule applies).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

#: The wire fields saved verbatim. `edge_attr`/`edge_index`/`node_coords`/`node_feat` are the
#: check's own inputs; the three offset arrays and `current_player` are what re-derive a single
#: graph's geometry from the fused batch without the producer.
_WIRE_FIELDS = (
    "node_feat", "node_coords", "edge_index", "edge_attr",
    "node_offsets", "edge_offsets", "legal_offsets", "legal_node_gather",
    "policy_dst_slot", "n_nodes_checksum", "n_stones", "window_center", "current_player",
)


def write_collate_dump(
    wire: Any, *, dump_dir: str | Path, context: dict[str, Any], error: BaseException,
) -> str | None:
    """Write the offending batch beside a JSON sidecar; return the sidecar path, or `None`.

    `None` means the dump could not be written and the caller must still raise — it is never a
    signal that nothing was wrong.

    Raises:
        Nothing. Every exception is caught: see the module docstring.
    """
    try:
        out = Path(dump_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = f"{context.get('round_id', 'noround')}_{int(time.time() * 1000)}"
        arrays: dict[str, Any] = {}
        for name in _WIRE_FIELDS:
            value = getattr(wire, name, None)
            if value is not None:
                arrays[name] = np.asarray(value)
        npz_path = out / f"collate_dump_{stamp}.npz"
        np.savez_compressed(npz_path, **arrays)

        offsets = arrays.get("edge_offsets")
        sidecar = {
            "finding": "F-816-37",
            "error_type": type(error).__name__,
            "error": str(error),
            "batch": npz_path.name,
            # The graph ids ARE the fused batch's membership: a global edge id lands in exactly
            # one graph's `[edge_offsets[g], edge_offsets[g+1])` span, so this list plus the
            # message's edge number is enough to name the offending GRAPH off-box.
            "n_graphs": int(getattr(wire, "n_graphs", 0) or 0),
            "edge_offsets": [int(x) for x in offsets] if offsets is not None else None,
            "contract_version": int(getattr(wire, "contract_version", -1) or -1),
            "builder_impl": int(getattr(wire, "builder_impl", -1) or -1),
            **context,
        }
        json_path = out / f"collate_dump_{stamp}.json"
        json_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")
        return str(json_path)
    except Exception:  # noqa: BLE001 — see the module docstring: the raise outranks the dump
        return None


__all__ = ["write_collate_dump"]
