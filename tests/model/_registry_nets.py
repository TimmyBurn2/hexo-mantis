"""The net classes the ban censuses name, derived from the arch registry rather than typed by hand."""
from __future__ import annotations

from mantis.model import ARCH_KINDS, build_net

#: A buried net stays banned by name: no registered kind builds it, so no derivation can find it.
BURIED_NETS: frozenset[str] = frozenset({"HexTacToeNet"})


def banned_net_names() -> frozenset[str]:
    """The class `build_net` returns for every registered arch kind, plus the buried nets."""
    built = {type(build_net(cls(in_dim=1, edge_dim=1))).__name__ for cls in ARCH_KINDS.values()}
    return frozenset(built) | BURIED_NETS
