"""Dump-on-fire for the graph-contract check: save the evidence before the raise.

The wire arrays are saved RAW — the same flat post-marshal buffers the check read, never a
reshaped view — because a reshape is a hypothesis about the layout and the layout is a suspect.
The graph ids and CSR offsets locate the offending edge inside its own graph rather than in the
fused batch, and the context records the round, phase and concurrency in force.

A DUMP MAY NEVER REPLACE THE RAISE: every failure here is swallowed and reported in the
returned path's place, because a diagnostic that turned a named contract failure into its own
`OSError` would destroy the evidence it exists to keep.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

#: The check's own inputs, plus the offset arrays that re-derive one graph's geometry from the
#: fused batch without the producer.
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
            # A global edge id lands in exactly one graph's `[edge_offsets[g], edge_offsets[g+1])`
            # span, so these offsets plus the message's edge number name the offending graph.
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
