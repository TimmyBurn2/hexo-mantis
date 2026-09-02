"""ADJ-WP12R-1 producer: the eval snapshot writes EXACTLY the keys its loader reads.

`write_model_snapshot` used to also write `"encoding"` and `"representation"` via
`getattr(..., None)` while `load_model_snapshot` read NEITHER — two written fields, zero
consumers (LAW-08), recorded through the silent-fallback shape the red-team greps for.
Deleted under the dead-weight law (R116). This oracle is what stops them coming back
unnoticed, and what would red if a future field is written without a reader.
"""
from __future__ import annotations

from pathlib import Path

import torch

from mantis.eval.snapshot import load_model_snapshot, write_model_snapshot
import pytest

from mantis.model import ARCH_KINDS, CnnArch, GnnArch, GnnArchV2, build_net

#: The payload contract: exactly what `load_model_snapshot` consumes, nothing else.
_EXPECTED_KEYS = {"state_dict", "arch"}


def _net():
    arch = CnnArch(board_size=19, in_channels=8, filters=8, res_blocks=1)
    net = build_net(arch)
    net.arch = arch
    net.eval()
    return net


def test_payload_carries_exactly_the_keys_the_loader_reads(tmp_path: Path) -> None:
    """THE PRODUCER. A written key outside this set is a LAW-08 gap by construction: the
    loader below is the only consumer, and it reads only these two."""
    path = tmp_path / "snap.pt"
    write_model_snapshot(_net(), path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert set(payload) == _EXPECTED_KEYS, (
        f"snapshot payload keys drifted from the loader's reads: {sorted(payload)}"
    )


def test_no_key_is_written_through_a_silent_getattr_fallback(tmp_path: Path) -> None:
    """The DEFECT SHAPE, not just the two field names. A model with no `.encoding` and an
    arch with no `.representation` must produce a payload identical to a model that has
    them — i.e. neither attribute may influence what is written. Before the fix this test
    would red: the payload recorded `None` for the missing attributes rather than failing,
    which is exactly the silent fallback LAW-11/R1 forbid.
    """
    plain = _net()
    tagged = _net()
    # The exact attribute the deleted `getattr(model, "encoding", None)` line read. The
    # arch half needs no counterpart: `CnnArch` is a FROZEN dataclass, so
    # `getattr(arch, "representation", None)` could only ever return the declared field or
    # None — it was unconditionally unread either way, which is the defect.
    tagged.encoding = "v6"

    a, b = tmp_path / "plain.pt", tmp_path / "tagged.pt"
    write_model_snapshot(plain, a)
    write_model_snapshot(tagged, b)

    ka = set(torch.load(a, map_location="cpu", weights_only=True))
    kb = set(torch.load(b, map_location="cpu", weights_only=True))
    assert ka == kb == _EXPECTED_KEYS, (
        f"an attribute on the model changed the payload key set: {ka} vs {kb}"
    )


def test_roundtrip_still_rebuilds_the_identical_net(tmp_path: Path) -> None:
    """NO UNRELATED CASUALTY (R81/R86): deleting the two unread fields must not disturb
    the load path. Weights and arch survive the roundtrip bit-for-bit."""
    path = tmp_path / "snap.pt"
    original = _net()
    write_model_snapshot(original, path)
    loaded = load_model_snapshot(path)

    assert loaded.arch == original.arch
    got, want = loaded.state_dict(), original.state_dict()
    assert set(got) == set(want)
    for key in want:
        assert torch.equal(got[key], want[key]), f"weight {key} changed across roundtrip"


# ── R330(e): the snapshot speaks the ONE arch-kind vocabulary ────────────────────────────
_TINY = {
    CnnArch: dict(board_size=19, in_channels=4, filters=8, res_blocks=1),
    GnnArch: dict(in_dim=11, edge_dim=5, hidden=8, num_layers=1, policy_hidden=8, value_hidden=8),
    GnnArchV2: dict(in_dim=11, edge_dim=5, hidden=8, num_layers=1, policy_hidden=8,
                    value_hidden=8),
}


@pytest.mark.parametrize("kind", sorted(ARCH_KINDS))
def test_every_arch_kind_in_the_vocabulary_round_trips_through_the_snapshot(tmp_path, kind):
    """AUDIT-1 F-16, closed at R330(e): this module carried a private two-row type table, so a
    `GnnArchV2` net could be trained and checkpointed but never snapshotted for the eval child —
    `write_model_snapshot` raised `unsupported arch type`. The table is now `ARCH_KINDS` itself,
    and this row walks the whole vocabulary so a fourth kind cannot lag the same way."""
    cls = ARCH_KINDS[kind]
    assert cls in _TINY, f"add tiny widths for {kind} — the vocabulary grew"
    arch = cls(**_TINY[cls])
    net = build_net(arch)
    net.arch = arch
    path = tmp_path / f"{kind}.pt"
    write_model_snapshot(net, path)
    back = load_model_snapshot(path)
    assert type(back.arch) is cls and back.arch == arch
    for key, value in net.state_dict().items():
        assert torch.equal(back.state_dict()[key], value), key
