"""`MonitorConfig` has ONE construction authority inside `src/`.

Inside `src/`, the only code that may construct a `MonitorConfig` is
`mantis.config.resolve.monitor.resolve_monitor_config`, which builds it from a VALIDATED schema
section by a pure 1:1 field copy; every other production consumer receives one. A bare
construction elsewhere silently substitutes the dataclass literals for whatever the operator
minted — armed in the config, absent in effect.

Tests MAY construct one directly: thresholds are their SUBJECT, and routing them through a
schema and a resolver would test the resolver instead. What a test may not rely on is a
PRODUCTION path falling back to a bare one. So: src/ constructs once, tests construct freely,
production never defaults — the first and third mechanically enforced.

Derived by AST, never by grep: a text search for `MonitorConfig(` misses `import ... as MC; MC()`
and hits comments and docstrings, and this repo has been bitten by both directions.
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "mantis"

#: The ONE legitimate construction site in `src/`, as a module path: a pure 1:1 field copy off a
#: validated `MonitorSchemaConfig`, with a field-name-equality mutation self-test behind it.
_THE_AUTHORITY = "config/resolve/monitor.py"



def _construction_sites() -> list[str]:
    """Every direct `MonitorConfig(...)` call under `src/`, resolved through import aliases."""
    sites: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        bound = {"MonitorConfig"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "MonitorConfig":
                        bound.add(alias.asname or alias.name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (isinstance(func, ast.Name) and func.id in bound) or (
                isinstance(func, ast.Attribute) and func.attr == "MonitorConfig"
            ):
                sites.append(f"{path.relative_to(SRC).as_posix()}:{node.lineno}")
    return sites


def test_only_the_resolver_constructs_a_MonitorConfig_in_src():
    """The class rule: a new bare construction in `src/` fails here and names itself."""
    offenders = [s for s in _construction_sites() if not s.startswith(_THE_AUTHORITY + ":")]
    assert not offenders, (
        "MonitorConfig is constructed outside the one authority at: " + ", ".join(offenders)
        + ". Production code receives a resolved config; it does not build one. A bare "
        "construction substitutes the dataclass literals for whatever the operator minted, "
        "which is F-816-24's defect repeated at a new site."
    )


def test_the_authority_itself_is_present_so_this_rule_is_not_vacuous():
    """Positive control: a rule with zero matches passes for the wrong reason, so the authority
    must be found where the rule says it is."""
    sites = _construction_sites()
    assert any(s.startswith(_THE_AUTHORITY + ":") for s in sites), (
        f"the one authority was not found in the census ({sites}) — either the resolver stopped "
        "constructing, or the AST walk stopped seeing it, and in both cases the rule above is inert"
    )


def test_no_production_path_falls_back_to_a_bare_config():
    """A required parameter cannot silently become a default.

    Derived from the signature, not the body: `StepCoordinator`'s `monitor_cfg` default used to
    be the last silent fallback in the chain, with `build_run_safety` already requiring it.
    """
    import inspect

    from mantis.train.coordinator.step import StepCoordinator

    param = inspect.signature(StepCoordinator.__init__).parameters["monitor_cfg"]
    assert param.default is inspect.Parameter.empty, (
        "`monitor_cfg` carries a default again — a default here is a bare MonitorConfig by another "
        "name, and the run's own composer requires the parameter one layer above"
    )


def test_the_rule_BITES_on_a_third_construction_site(tmp_path):
    """Self-test: an allowlist rule that has never been shown to reject anything is
    indistinguishable from a rule that accepts everything."""
    planted = tmp_path / "mantis"
    (planted / "somewhere").mkdir(parents=True)
    (planted / "somewhere" / "new_consumer.py").write_text(
        "from mantis.monitor.config import MonitorConfig\n"
        "cfg = MonitorConfig()\n", encoding="utf-8",
    )
    import test_monitor_config_single_authority as mod  # self-import: reuse the real walker

    original, mod.SRC = mod.SRC, planted
    try:
        offenders = [
            s for s in mod._construction_sites()
            if not s.startswith(mod._THE_AUTHORITY + ":")
        ]
    finally:
        mod.SRC = original
    assert offenders == ["somewhere/new_consumer.py:2"], (
        f"a planted third construction site was not rejected: {offenders}"
    )
