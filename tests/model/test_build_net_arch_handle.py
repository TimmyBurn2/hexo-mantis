"""⊕ WPAX Phase P ORACLE — O-8: `build_net` carries the declared arch (DESIGN_P §13.2, M-7).

RED-at-EXECUTION, not at collection: TD-2 is a two-line MODIFICATION to
`src/mantis/model/build.py`, so there is no import anchor. At HEAD `build_net(arch)` returns
a module with **no** `.arch`, and both drives below fail with `AttributeError: 'GnnNet'
object has no attribute 'arch'` / `'HexTacToeNet' object has no attribute 'arch'`. That
measured absence IS the RED.

The defect this file is the ONLY witness to: **one half of a stated convention was never
implemented.** `eval/snapshot.py:45-51` documents the arch-travels-with-the-model convention
and RAISES without it (`getattr(model, "arch", None)`, `:48`), and `eval/snapshot.py:82`
implements the LOAD side (`model.arch = arch`). Nothing implements the BUILD side, so the
terminal eval (`eval/pipeline.py:312`) and the anchor snapshot (`:316`) both die on the
first snapshot of a freshly-built net. Phase P's own tool cannot validate this — TD-1 blocks
the boot long before it — which is precisely why the producer test is a unit-level direct
drive and not a drive-through (REVIEW_DESIGN_P ruling 2a).

N-7 / `repo_design.md:121-124`, stated here because a future reader will cite the ban
against this file: what is banned is DERIVING arch metadata from a live module's structure
(the deleted `model_representation` sniff, and the hyperparameter reads
`tests/model/test_arch_ban.py` enumerates as `_ARCH_ATTRS`). What this oracle pins is the
opposite direction — the declared dataclass instance, the very object §3 says arch travels
on, carried as a handle. The gate was re-run against the candidate patch and a planted
control this run and does not bite (DESIGN_P §3.3 TD-2 (i)).

N-6, measured and deliberately NOT asserted below: `copy.deepcopy(net)` KEEPS `.arch` but
BREAKS `is` identity, while `.to(...)` preserves it. `is` is the correct assertion **at
`build_net`** — a copy of the arch would be a second authority for the run's identity — but
**no downstream consumer may rely on `is`**; presence and equality are all a consumer gets.
"""
from __future__ import annotations

from mantis.encoding import lookup
from mantis.model import CnnArch, GnnArch, arch_from_spec_and_config, build_net


def _archs():
    """One CnnArch and one GnnArch, both built the way production builds them — through
    `arch_from_spec_and_config` off a REGISTERED encoding, never hand-constructed (LAW-11).
    """
    grid = arch_from_spec_and_config(lookup("v6"), {})
    graph = arch_from_spec_and_config(lookup("gnn_axis_v1"), {})
    assert isinstance(grid, CnnArch) and isinstance(graph, GnnArch)
    return {"v6": grid, "gnn_axis_v1": graph}


def test_build_net_carries_the_declared_arch_dataclass_as_a_handle() -> None:
    """`build_net(arch).arch is arch` for BOTH representations.

    Identity, not equality: an implementation that stores a copy, a `replace()`, or a
    re-derived arch would satisfy equality while creating exactly the second arch authority
    repo_design §3 exists to forbid. Both arms are driven because a one-armed fix (grid
    only, or graph only) is the shape that leaves `configs/run5.yaml` — a `graph` run —
    dying at `eval/snapshot.py:48` after a green test suite.
    """
    for name, arch in _archs().items():
        net = build_net(arch)
        assert getattr(net, "arch", None) is not None, (
            f"{name}: build_net must attach the declared arch — `eval/snapshot.py:48` reads "
            "`getattr(model, 'arch', None)` and RAISES when it is absent, so the terminal "
            "eval and the anchor snapshot cannot run without it (DESIGN_P §3.3 TD-2)"
        )
        assert net.arch is arch, (
            f"{name}: the attached arch must be THE declared instance, not a copy or a "
            f"re-derivation; got {net.arch!r} (id differs from the passed {arch!r})"
        )


def test_the_arch_handle_never_enters_the_state_dict_or_the_module_registries() -> None:
    """LAW-12's boundary: checkpoints save `state_dict()` payloads
    (`checkpoints.py:302,340`; `snapshot.py:65`), never the module — so the handle must land
    in `__dict__` and in NONE of `_parameters` / `_buffers` / `_modules`. Measured this run:
    `state_dict()` is byte-identical before and after the fix, 50 keys unchanged.

    Without this pin, a `register_buffer`-shaped implementation would satisfy the identity
    test above and silently change every checkpoint's key set.
    """
    for name, arch in _archs().items():
        net = build_net(arch)
        assert "arch" in vars(net), f"{name}: the handle belongs in __dict__"
        for registry in ("_parameters", "_buffers", "_modules"):
            assert "arch" not in getattr(net, registry, {}), (
                f"{name}: 'arch' must not be registered in {registry} — that would change "
                "the state-dict key set and break LAW-12's byte-parity"
            )
        assert not any(key == "arch" or key.startswith("arch.")
                       for key in net.state_dict()), (
            f"{name}: the arch handle must not appear in state_dict(); keys must be "
            "unchanged by TD-2"
        )
