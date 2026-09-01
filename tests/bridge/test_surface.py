"""WP7 surface-presence oracle: the full assembled `mantis._engine` face.

Pins the WP0 close-out re-prove (O16) inventory — 11 pyclasses + 4 free fns +
3 module fns + the `WireAlreadyConsumed` exception — the F-42 `__module__`
decision (O12, pinned: every pyclass reports `mantis._engine`, never
`builtins`), and the NEW-BUILD `all_specs()` binding (O7).
"""
from mantis import _engine

PYCLASSES = [
    "Board",
    "RegistrySpec",
    "MCTSTree",
    "TacticalSolver",
    "InferenceBatcher",
    "GraphWire",
    "SelfPlayRunnerConfig",
    "SelfPlayRunner",
    "ReplayBuffer",
    "HexgBuffer",
    "GraphTargets",
]
FREE_FNS = [
    "verify_edge_geometry",
    "apply_symmetries_batch",
    "mcts_pool_overflow_count",
    "take_mcts_pool_overflow_count",
]
MODULE_FNS = ["all_specs", "registry_sha", "registry_sha_hex"]
REGISTERED_NAMES = {"v6", "v6w25", "v6_live2_ls", "gnn_axis_v1", "gnn_axis_r8"}


def test_all_11_pyclasses_present():
    missing = [c for c in PYCLASSES if not isinstance(getattr(_engine, c, None), type)]
    assert not missing, f"missing pyclasses: {missing}"


def test_all_4_free_fns_present():
    missing = [f for f in FREE_FNS if not callable(getattr(_engine, f, None))]
    assert not missing, f"missing free fns: {missing}"


def test_all_3_module_fns_present():
    missing = [f for f in MODULE_FNS if not callable(getattr(_engine, f, None))]
    assert not missing, f"missing module fns: {missing}"


def test_wire_already_consumed_exception_present():
    assert issubclass(_engine.WireAlreadyConsumed, Exception)


def test_f42_every_pyclass_module_is_engine():
    """F-42 (pinned): explicit `module = "mantis._engine"` on every pyclass —
    never the pyo3 default `builtins`."""
    wrong = {
        c: getattr(_engine, c).__module__
        for c in PYCLASSES
        if getattr(_engine, c).__module__ != "mantis._engine"
    }
    assert not wrong, f"F-42 violation (module != mantis._engine): {wrong}"


def test_all_specs_binding_matches_registered_set():
    """O7: `all_specs()` yields exactly the 4 TOML-registered names as
    RegistrySpec instances."""
    specs = _engine.all_specs()
    assert all(isinstance(s, _engine.RegistrySpec) for s in specs)
    assert {s.name for s in specs} == REGISTERED_NAMES


def test_registry_sha_shapes():
    raw = _engine.registry_sha()
    hexed = _engine.registry_sha_hex()
    assert isinstance(raw, bytes) and len(raw) == 32
    assert isinstance(hexed, str) and len(hexed) == 64
    assert bytes.fromhex(hexed) == raw
