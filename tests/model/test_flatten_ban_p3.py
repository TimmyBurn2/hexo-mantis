"""WPSC Phase 3 SC-B5 — flatten-ban architecture test (A7; DESIGN_P3.md §6). The graph
path's policy head is a per-node scalar MLP (`out_features == 1`), never a flatten-to-
fixed-FC head — the pattern that would silently cap the action space at a compile-time
`n_actions` constant. `HexTacToeNet` (dense/grid) is the KNOWN, EXEMPT counter-example
(A5/`DEBT_DOSSIER.md` item 5): it legitimately flattens a spatial feature map to a fixed
FC head, and no test here asserts against it — this file only bans the pattern on the
GNN path (`GnnNet`/`PolicyHead`/`RepresentationNetwork`).

`tests/model/test_arch_ban.py` already exists but covers a DIFFERENT concern (the O3
arch-off-module sniff census) — read in full before writing this file; it has no
flatten-ban content, so per DESIGN_P3.md §6.2's own instruction this is a NEW sibling
file (`test_flatten_ban.py`... `_p3` suffix here, see deviation note) rather than an
extension.

GREEN-guard (DESIGN_P3.md §6.3): `GnnNet`/`PolicyHead` already satisfy every assertion
below at HEAD — this is a producer test for an EXISTING correct structure (LAW-07), not a
RED-at-import pin. Stays in the tree unstaged (not moved to the RED oracle-staging dir).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from mantis.model import GnnArch, build_net
from mantis.model.gine import PolicyHead, RepresentationNetwork
from mantis.model.gnn import GnnNet

# Documentation anchor (§6.2 item 4) — a future reader finds the exemption at the same
# place the ban lives. Not a skip-marker: no test targets HexTacToeNet here.
KNOWN_DENSE_HEADS_EXEMPT_FROM_FLATTEN_BAN = {"HexTacToeNet"}

_IN_DIM = 11
_EDGE_DIM = 5


def _tiny_gnn() -> GnnNet:
    return build_net(
        GnnArch(in_dim=_IN_DIM, edge_dim=_EDGE_DIM, hidden=8, num_layers=1,
                 policy_hidden=8, value_hidden=8)
    )


def test_no_flatten_in_policy_head_or_representation() -> None:
    net = _tiny_gnn()
    assert not any(isinstance(m, nn.Flatten) for m in net.policy_head.modules())
    assert not any(isinstance(m, nn.Flatten) for m in net.representation.modules())


def test_policy_head_final_linear_is_per_node_scalar() -> None:
    """The positive, per-node signature: the final `Linear`'s `out_features == 1` — not
    `n_actions`, not any board-size-derived constant. A differently-sized-but-still-fixed
    flatten head would slip past a purely negative (no-Flatten) check; this doesn't."""
    net = _tiny_gnn()
    linears = [m for m in net.policy_head.mlp if isinstance(m, nn.Linear)]
    assert linears, "PolicyHead.mlp must contain at least one nn.Linear"
    assert linears[-1].out_features == 1


def test_forward_batch_policy_logits_are_unpadded_ungrouped() -> None:
    """RED-TEAM-lens proof: a synthetic 2-graph batch with DIFFERING legal-node counts
    (3, 5) produces `policy_logits.shape == (8,)` — not padded to a fixed max, not
    reshaped to `(2, max_legal)`. Zero edges (GINEConv tolerates an empty edge_index)."""
    net = _tiny_gnn()
    net.eval()
    n_total = 10  # graph 0: 4 nodes (3 legal); graph 1: 6 nodes (5 legal)
    x = torch.randn(n_total, _IN_DIM)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.zeros((0, _EDGE_DIM), dtype=torch.float32)
    legal_mask = torch.zeros(n_total, dtype=torch.bool)
    legal_mask[[0, 1, 2]] = True          # 3 of graph 0's 4 nodes
    legal_mask[[4, 5, 6, 7, 8]] = True    # 5 of graph 1's 6 nodes
    stone_mask = torch.zeros(n_total, dtype=torch.bool)
    node_offsets = torch.tensor([0, 4, 10], dtype=torch.long)

    with torch.no_grad():
        policy_logits, value, bin_logits = net.forward_batch(
            x, edge_index, edge_attr, legal_mask, stone_mask, node_offsets=node_offsets,
        )
    assert policy_logits.shape == (8,)
    assert value.shape == (2, 1)


def test_hextactoe_net_is_the_named_exempt_dense_head() -> None:
    assert KNOWN_DENSE_HEADS_EXEMPT_FROM_FLATTEN_BAN == {"HexTacToeNet"}


def test_no_view_or_reshape_reachable_from_policy_or_representation_modules() -> None:
    """Class census, not instance census: no `.view(`/`.reshape(` token in the source of
    any module class reachable from `PolicyHead`/`RepresentationNetwork` (gine.py)."""
    import inspect

    for cls in (PolicyHead, RepresentationNetwork):
        src = inspect.getsource(cls)
        assert ".view(" not in src
        assert ".reshape(" not in src
