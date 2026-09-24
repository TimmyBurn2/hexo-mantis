"""The tests' one fused-graph cap pair, resolved off the dev template config's inference block."""
from __future__ import annotations

from pathlib import Path

from mantis.config.loader import load_config
from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec, resolve_fused_graph_caps

_REPO = Path(__file__).resolve().parents[1]

#: The template's NON-BINDING-BY-CONSTRUCTION pair, read through the production resolve path, so
#: a re-mint of dev_example moves every harness that imports this and no site restates literals.
CAPS: FusedGraphCapsSpec = resolve_fused_graph_caps(
    load_config(_REPO / "configs" / "dev_example.yaml").model_dump()
)

#: The same pair as a plain `inference.fused_graph_caps` block, for sites that build a mapping.
CAPS_DICT: dict[str, int] = {
    "max_fused_edges": CAPS.max_fused_edges,
    "max_fused_nodes": CAPS.max_fused_nodes,
}
