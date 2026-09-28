"""The tactics block a tree arms: a well-formed one arms and None disarms; a malformed one is refused by name."""
from __future__ import annotations

from typing import Any

import pytest

from mantis._engine import MCTSTree

_LEAF: dict[str, Any] = {"kind": "strict_turn", "leaf_turns": 2, "leaf_nodes": 64, "root_turns": 8,
                         "root_nodes": 0, "audit": None}
_AUDIT: dict[str, Any] = {"turns": 2, "nodes": 256, "k": 4, "m": 2, "total_nodes": 4096, "mode": "hold"}


def test_a_well_formed_block_arms_and_none_disarms() -> None:
    tree = MCTSTree()
    tree.configure_tactics({**_LEAF, "audit": {**_AUDIT, "mode": "inverted"}})
    assert tree.tactics_armed
    tree.configure_tactics(None)
    assert not tree.tactics_armed


@pytest.mark.parametrize(("block", "names"), [
    ({k: v for k, v in _LEAF.items() if k != "audit"}, "keys"),
    ({**_LEAF, "forced_block": True}, "keys"),
    ({**_LEAF, 1: 2}, "a key is not a string"),
    ({**_LEAF, "kind": "per_stone"}, "kind"),
    ({**_LEAF, "kind": 3}, "kind is not a string"),
    ({**_LEAF, "leaf_turns": 0}, "leaf_turns"),
    ({**_LEAF, "leaf_nodes": True}, "leaf_nodes is not an integer"),
    ({**_LEAF, "root_nodes": 1.5}, "root_nodes is not an integer"),
    ({**_LEAF, "audit": 7}, "audit is not a dict"),
    ({**_LEAF, "audit": {**_AUDIT, "mode": "fold"}}, "mode"),
    ({**_LEAF, "audit": {**_AUDIT, "mode": None}}, "mode is not a string"),
    ({**_LEAF, "audit": {**_AUDIT, "k": 0}}, "k=0"),
])
def test_a_malformed_block_is_refused_by_name_and_arms_nothing(block: dict[Any, Any], names: str) -> None:
    tree = MCTSTree()
    with pytest.raises(ValueError, match=names):
        tree.configure_tactics(block)
    assert not tree.tactics_armed


def test_a_block_that_is_not_a_dict_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        MCTSTree().configure_tactics([("kind", "strict_turn")])  # type: ignore[arg-type]
