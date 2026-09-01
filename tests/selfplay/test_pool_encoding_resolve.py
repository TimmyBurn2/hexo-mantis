"""⊕ D-01 — pool encoding resolve, per registered encoding (WP-SP).

Written oracle-first against the dispatcher's old-side capture (#C3a, wp/WPSP/CAPTURE_LOG.md)
BEFORE any port code. RED at import until IMPL writes `mantis.selfplay.hparams`.

This file carries D-01 ONLY. The rest of Suite D (D-02 … D-17) is IMPL-written; its capture
rows live in the same `wp/WPSP/oldside/` bank and may be promoted alongside them.

Q3 ruling this pins: the resolver is ALREADY the kept-plane authority, so `n_kept_planes` and
`kept_plane_indices` must come from the bound spec — the old module-level `KEPT_PLANE_INDICES`
const (v6-only, and a live hazard on a 10-channel spec) is not ported.
"""
from __future__ import annotations

import pytest

from mantis.selfplay.hparams import resolve_pool_encoding

# THE ENCODINGS THE OLD-SIDE CAPTURE COVERS — not the live registered set, and the two have
# diverged: R328(b) registered `gnn_axis_r8`, which post-dates the capture and therefore has no
# golden to be compared against. This tuple is a property of the FIXTURE, so it does not grow
# when the registry does; `tests/bridge/test_surface.py` is where the live set is pinned.
REGISTERED = ("v6", "v6w25", "v6_live2_ls", "gnn_axis_v1")


@pytest.mark.parametrize("name", REGISTERED)
def test_resolve_encoding_per_registered_name(encoding_resolve_golden, name):
    """D-01 — PASS iff `resolve_pool_encoding({"encoding": name})` reproduces the captured
    old-side tuple for all four registered encodings: encoding_name, board_size, trunk_size,
    n_kept_planes, plus the bound spec's representation / policy_logit_count /
    node_feat_dim / edge_feat_dim / kept_plane_indices. FAIL on ANY field = the pool would
    size its feature and policy buffers from a different spec than the one the encoder used —
    the class of mismatch that produces silently-wrong training data rather than a crash."""
    golden = encoding_resolve_golden["encodings"][name]
    assert golden["outcome"] == "ok", f"capture says {name} did not resolve old-side"

    resolved = resolve_pool_encoding({"encoding": name}, arch=None)

    assert resolved.encoding_name == golden["encoding_name"] == name
    assert int(resolved.board_size) == golden["board_size"]
    assert int(resolved.trunk_size) == golden["trunk_size"]
    assert int(resolved.n_kept_planes) == golden["n_kept_planes"]

    spec = resolved.registry_spec
    assert str(spec.representation) == golden["spec_representation"]
    assert int(spec.policy_logit_count) == golden["spec_policy_logit_count"]
    assert [int(i) for i in spec.kept_plane_indices] == golden["spec_kept_plane_indices"], (
        "kept_plane_indices must come from the bound spec (Q3: the resolver is the single "
        "kept-plane authority; no selfplay-local const)"
    )

    node_dim = getattr(spec, "node_feat_dim", None)
    edge_dim = getattr(spec, "edge_feat_dim", None)
    assert (None if node_dim is None else int(node_dim)) == golden["spec_node_feat_dim"]
    assert (None if edge_dim is None else int(edge_dim)) == golden["spec_edge_feat_dim"]


def test_resolved_kept_planes_match_indices_length(encoding_resolve_golden):
    """D-01 (consistency arm) — PASS iff `n_kept_planes` equals `len(kept_plane_indices)` for
    every registered encoding, as the capture shows (v6/v6w25 8, v6_live2_ls 4, graph 0).
    FAIL = the plane COUNT and the plane INDEX LIST come from different places, which is how
    a 4-plane spec ends up slicing 8 planes out of a checkpoint."""
    for name in REGISTERED:
        golden = encoding_resolve_golden["encodings"][name]
        assert golden["n_kept_planes"] == len(golden["spec_kept_plane_indices"])

        resolved = resolve_pool_encoding({"encoding": name}, arch=None)
        assert int(resolved.n_kept_planes) == len(resolved.registry_spec.kept_plane_indices), (
            f"{name}: n_kept_planes disagrees with the bound spec's kept_plane_indices"
        )
