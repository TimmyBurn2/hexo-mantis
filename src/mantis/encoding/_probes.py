"""Shared state-dict key probe for encoding detection.

The GRAPH state-dict marker. A graph (GNN) state dict's unambiguous signature is the GNN
representation trunk's first Linear (`input_proj`). Canonical single definition — resolvers.py
and any future detector import from here, never inline the string. The dense first-conv /
policy-fc probes went with the grid path (R346(f)).
"""
from __future__ import annotations

GNN_GRAPH_MARKER_KEY: str = "representation.input_proj.weight"
