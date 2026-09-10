"""`build_net` carries the declared arch as a handle.

`eval/snapshot.py` documents the arch-travels-with-the-model convention, RAISES without it and
implements the LOAD side; nothing implemented the BUILD side, so the terminal eval and the anchor
snapshot both died on the first snapshot of a freshly-built net.

What the arch ban forbids is DERIVING arch metadata from a live module's structure; this pins the
opposite direction — the declared dataclass instance carried as a handle. `is` is the correct
assertion AT `build_net` (a copy would be a second authority for the run's identity), but no
downstream consumer may rely on it: `copy.deepcopy(net)` keeps `.arch` and breaks identity.
"""
from __future__ import annotations

from mantis.encoding import lookup
from mantis.model import GnnArch, arch_from_spec_and_config, build_net


def _archs():
    """Every registered encoding's arch, built off a REGISTERED encoding, never hand-constructed."""
    archs = {name: arch_from_spec_and_config(lookup(name), {})
             for name in ("gnn_axis_v1", "gnn_axis_r8")}
    assert all(isinstance(a, GnnArch) for a in archs.values())
    return archs


def test_build_net_carries_the_declared_arch_dataclass_as_a_handle() -> None:
    """`build_net(arch).arch is arch` for BOTH representations.

    Identity, not equality: a stored copy, a `replace()` or a re-derived arch satisfies equality
    while creating a second arch authority.
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
    """The handle lands in `__dict__` and in NONE of `_parameters`/`_buffers`/`_modules`.

    Checkpoints save `state_dict()` payloads, never the module, so a `register_buffer`-shaped
    implementation would satisfy the identity test above and silently change every checkpoint's
    key set.
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
