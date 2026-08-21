"""R292(b) class-wide — `MonitorConfig` has ONE construction authority inside `src/`.

THE CLASS, AND WHY IT NEEDED A SEPARATE ACT FROM THE SUPERVISOR FIX. `F-816-24` was one instance:
`monitor/supervise.py` built a bare `MonitorConfig()` and used its dataclass literals as flag
defaults, so four minted keys reached no process. R292(b) SPLIT the work — that packet built the
mechanism at its own site and this one extends it across the class, after that merge, precisely so
the extension has a landed mechanism to extend rather than a second design.

WHAT "ONE AUTHORITY" MEANS HERE, stated as the rule the test enforces: inside `src/`, the only
code that may construct a `MonitorConfig` is `mantis.config.resolve.monitor.resolve_monitor_config`,
which builds it from a VALIDATED schema section by a pure 1:1 field copy. Every other production
consumer receives one. A bare construction anywhere else silently substitutes 29 dataclass literals
for whatever the operator minted — armed in the config, absent in effect.

THE RULE FOR TESTS, which R292(b) also asks for, and it is deliberately weaker than the src/ rule:
a test MAY construct a `MonitorConfig` directly, with or without kwargs. Tests are where the
thresholds are the SUBJECT — `monitor/rules.py`'s fire/no-fire rows need a config whose values they
chose, and routing them through a schema and a resolver would test the resolver instead. What tests
may NOT do is rely on a PRODUCTION path falling back to a bare one; that fallback is what this file
removes. The boundary is therefore "src/ constructs once, tests construct freely, production never
defaults" — and only the first and third are mechanically enforced, because the second is not a
defect.

DERIVED BY AST, NEVER BY GREP (R296(f)). A text search for `MonitorConfig(` misses
`from … import MonitorConfig as MC; MC()` and `import … as m; m.MonitorConfig()`, and hits comments
and docstrings — this repo has already been bitten by both directions of that, which is why the
convention exists.
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "mantis"

#: The ONE legitimate construction site in `src/`, as a module path. It is the resolver: a pure
#: 1:1 field copy off a validated `MonitorSchemaConfig`, with a field-name-equality mutation
#: self-test behind it (`tests/config/test_monitor_schema.py`).
_THE_AUTHORITY = "config/resolve/monitor.py"

#: The ONE remaining exception, and it is FILED rather than tolerated: `F-816-29`. Making
#: `StepCoordinator.monitor_cfg` required — which is what R292(b) actually wants — forces an edit
#: to `tests/train/test_periodic_checkpoint.py`, a FROZEN oracle that constructs the coordinator
#: without it. That edit needs an R43 grant. The site is listed here WITH its row number so the
#: rule still bites on any THIRD site: an allowlist that grows silently is not a rule, and an
#: exception without a row is a defect wearing a comment.
_FILED_EXCEPTION = "train/coordinator/step.py"


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


def test_no_construction_site_in_src_beyond_the_resolver_and_the_filed_one():
    """The class rule. A new bare construction in `src/` fails here and names itself.

    The name says "beyond the resolver AND THE FILED ONE" rather than "only the resolver",
    because the assertion allows two sites and a name that claimed one would be the overclaiming
    class this repo keeps finding. When `F-816-29`'s grant lands, the exception goes and the name
    should shrink with it.
    """
    allowed = (_THE_AUTHORITY + ":", _FILED_EXCEPTION + ":")
    offenders = [s for s in _construction_sites() if not s.startswith(allowed)]
    assert not offenders, (
        "MonitorConfig is constructed outside the one authority at: " + ", ".join(offenders)
        + ". Production code receives a resolved config; it does not build one. A bare "
        "construction substitutes the dataclass literals for whatever the operator minted, "
        "which is F-816-24's defect repeated at a new site."
    )


def test_the_authority_itself_is_present_so_this_rule_is_not_vacuous():
    """A rule with zero matches passes for the wrong reason. This is the positive control: the
    authority must be found where the rule says it is, or the census above is measuring nothing."""
    sites = _construction_sites()
    assert any(s.startswith(_THE_AUTHORITY + ":") for s in sites), (
        f"the one authority was not found in the census ({sites}) — either the resolver stopped "
        "constructing, or the AST walk stopped seeing it, and in both cases the rule above is inert"
    )


def test_the_one_remaining_fallback_is_the_FILED_one_and_has_not_moved():
    """The other half: a required parameter cannot silently become a default.

    `StepCoordinator` used to take `monitor_cfg: MonitorConfig | None = None` and fall back to a
    bare one — the last silent default in the chain, with `build_run_safety` already requiring it
    one layer up. Derived from the signature, not from the body.
    """
    import inspect

    from mantis.train.coordinator.step import StepCoordinator

    param = inspect.signature(StepCoordinator.__init__).parameters["monitor_cfg"]
    assert param.default is None, (
        "`monitor_cfg`'s default changed. It is `None` and the fallback is live BY DISCLOSURE — "
        "F-816-29: making it required needs an R43 grant to a frozen oracle. If this is now "
        "`empty`, the grant landed and both this row and the _FILED_EXCEPTION above should go."
    )


def test_the_rule_BITES_on_a_third_construction_site(tmp_path):
    """LAW-07 self-test. The claim "a third site still fails" is worth exactly as much as a
    demonstration of it — an allowlist rule that has never been shown to reject anything is
    indistinguishable from a rule that accepts everything.
    """
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
            if not s.startswith((mod._THE_AUTHORITY + ":", mod._FILED_EXCEPTION + ":"))
        ]
    finally:
        mod.SRC = original
    assert offenders == ["somewhere/new_consumer.py:2"], (
        f"a planted third construction site was not rejected: {offenders}"
    )
