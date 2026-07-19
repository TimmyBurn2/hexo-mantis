"""O3b — promoted-anchor killed-branch absence (MF1).

For each promoted anchor artifact whose captured key set is available, assert it
carries ZERO `cluster_pool.` / `global_encoder.` / `gpool_bias_branch.` keys AND
every key ⊆ the stripped arch's constructed (dist65-superset) key set for that
encoding. This turns "the killed arms were never promoted" into a checked fact.

Only `bootstrap_model_v6_live2.pt` (v6_live2_ls) is present in the committed capture
(0 killed keys / 147 total, scalar head). The other three anchors are operator-box
only → O3b INCONCLUSIVE for them (recorded, NOT a pass); the §f WP10/WP11 loader
fallback (reject killed prefixes, never reconstruct) governs.
"""
from __future__ import annotations

from pathlib import Path

from mantis.encoding import lookup
from mantis.model import arch_from_spec_and_config, build_net

_ANCHOR_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "value_probes" / "anchor_keys"
_KILLED_PREFIXES = ("cluster_pool.", "global_encoder.", "gpool_bias_branch.")

# anchor file stem -> registered encoding it is consumed as.
_ANCHORS = {
    "v6_live2": "v6_live2_ls",   # PRESENT (pre-verified clean)
    "v6": "v6",                  # operator-box only -> INCONCLUSIVE
    "v6w25": "v6w25",            # operator-box only -> INCONCLUSIVE
    "gnn_axis_v1": "gnn_axis_v1",  # operator-box only -> INCONCLUSIVE
}


def _constructed_dist65_superset_keys(enc: str) -> set[str]:
    spec = lookup(enc)
    cfg = {} if spec.representation == "graph" else {"value_head_type": "dist65"}
    return set(build_net(arch_from_spec_and_config(spec, cfg)).state_dict().keys())


def test_promoted_anchors_have_no_killed_keys_and_are_covered() -> None:
    gated = 0
    for stem, enc in _ANCHORS.items():
        f = _ANCHOR_DIR / f"{stem}.txt"
        if not f.is_file():
            continue  # INCONCLUSIVE — anchor is operator-box only (§f fallback governs)
        keys = set(f.read_text().split())
        assert keys, f"{stem}: empty anchor key set"
        killed = [k for k in keys if k.startswith(_KILLED_PREFIXES)]
        assert not killed, f"{stem}: DIRTY anchor carries killed-branch keys {killed}"
        superset = _constructed_dist65_superset_keys(enc)
        uncovered = keys - superset
        assert not uncovered, f"{stem}: keys not covered by stripped arch: {sorted(uncovered)}"
        gated += 1
    # At least the pre-verified v6_live2 anchor must be present and gated.
    assert gated >= 1, "the v6_live2 promoted-anchor key set must be committed and gated"


def test_v6_live2_anchor_is_scalar_head() -> None:
    """The present anchor is the scalar baseline (no value_fc2_bins) — matches capture #10."""
    keys = set((_ANCHOR_DIR / "v6_live2.txt").read_text().split())
    assert not any(k.startswith("value_fc2_bins") for k in keys)
    # equals the constructed v6_live2_ls SCALAR key set exactly.
    scalar_keys = set(
        build_net(arch_from_spec_and_config(lookup("v6_live2_ls"), {})).state_dict().keys()
    )
    assert keys == scalar_keys
