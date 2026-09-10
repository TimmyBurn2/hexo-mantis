"""The bot resolver's refusal surface.

The ENV-KEY channel through which all three external kinds were refused is DELETED: the authority
for where the sealbot engine lives is `vendor/pins.toml` + `make vendor`, and for the deleted
kinds the key was a silent-arming surface with nothing behind it.

Each row is the only witness to one defect: a sealbot skip that does not say which STEP is
missing, where three environments must give three PAIRWISE DISTINCT reasons each naming the
command that fixes it; a ruled skip that reads like a dispatcher shortfall, its grounds being PER
RUNG; the env channel surviving, observed both BEHAVIOURALLY and by SOURCE, since a dead key
still reads as an arming surface and a live key producing equal strings is a disguised channel;
and a host path or endpoint entering `bots/`, scoped to the token classes the sibling census does
not cover.

>300 justify: one resolver, one file. Every row asserts something about the SAME refusal surface,
and the point of the pairwise-distinctness rows is that a reader can tell the ruled skips from
the environment-state refusals — an oracle seeing one class at a time cannot assert that.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest

from mantis.bots.protocol import RungUnresolvable

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_BOTS_SRC = _SRC / "mantis" / "bots"

#: The three deleted env keys, named here in the ORACLE because after the rewrite there is
#: nowhere in `src/` left to read them from.
_DEAD_ENV_KEYS = {
    "sealbot": "MANTIS_BOT_SEALBOT",
}

#: The two commands a skip reason must name. Not host paths and not endpoints: `make vendor` is
#: the repo's ONE vendoring mechanism, and the build invocation runs inside the gitignored tree.
_VENDOR_CMD = "make vendor"
_BUILD_CMD = "build_ext --inplace"

_RESOLVED = "<resolved>"


def _reason_or_resolved(kind: str, *, depth: int | None) -> str:
    """Return the outcome CLASS of one `resolve_bot` call, as a comparable string.
    Environment-robust on purpose: where the extension is built `sealbot` RESOLVES, and an oracle
    that hard-required a raise would red for an environment reason rather than its own."""
    from mantis.bots.resolve import resolve_bot

    try:
        resolve_bot(kind, depth=depth, opponent_sims=128)
    except RungUnresolvable as exc:
        return exc.reason
    return _RESOLVED


def _no_vendor_root(monkeypatch: pytest.MonkeyPatch) -> None:
    import mantis.bots.sealbot as sealbot_mod

    monkeypatch.setattr(sealbot_mod, "find_vendor_root", lambda: None)


# Three environments, three distinct reasons, each naming its own missing step.
def test_sealbot_refusal_reasons_are_pairwise_distinct_across_three_environments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """All three arms plus the distinctness assertion, in ONE function so the `[unreached]`
    labels stay valid: if the loader stops raising, arm (a) fails at block exit and the rest
    never execute. Post-conditions sit OUTSIDE every block."""
    import mantis.bots.sealbot as sealbot_mod
    from mantis.bots.resolve import resolve_bot

    # (a) no vendor root at all -> name the fetch step.
    monkeypatch.setattr(sealbot_mod, "find_vendor_root", lambda: None)
    with pytest.raises(RungUnresolvable) as absent_exc:
        resolve_bot("sealbot", depth=5, opponent_sims=128)
    reason_no_vendor = absent_exc.value.reason
    assert absent_exc.value.rung == "sealbot"
    assert _VENDOR_CMD in reason_no_vendor, reason_no_vendor

    # (b) vendor root present, extension absent -> name the BUILD step; `tmp_path` holds no
    # built extension, which is the whole of the condition.
    monkeypatch.setattr(sealbot_mod, "find_vendor_root", lambda: tmp_path)
    with pytest.raises(RungUnresolvable) as unbuilt_exc:
        resolve_bot("sealbot", depth=5, opponent_sims=128)
    reason_no_build = unbuilt_exc.value.reason
    assert _BUILD_CMD in reason_no_build, reason_no_build

    # (c) the loader itself raised -> carry the underlying failure's repr. A loader that
    # swallowed it would report "not built" for an ABI mismatch.
    def _explode() -> tuple[Any, Any]:
        raise ImportError("undefined symbol: _ZTIN8pybind116detail13type_casterE")

    monkeypatch.setattr(sealbot_mod, "load_sealbot_modules", _explode)
    with pytest.raises(RungUnresolvable) as loader_exc:
        resolve_bot("sealbot", depth=5, opponent_sims=128)
    reason_loader = loader_exc.value.reason
    assert "undefined symbol" in reason_loader, reason_loader

    reasons = [reason_no_vendor, reason_no_build, reason_loader]
    assert len(set(reasons)) == 3, (
        f"the three sealbot refusal environments must be PAIRWISE DISTINGUISHABLE from the "
        f"reason string alone; got {reasons}"
    )


@pytest.mark.parametrize(
    ("arm", "must_contain", "must_not_contain"),
    [("vendor_absent", _VENDOR_CMD, _BUILD_CMD), ("build_absent", _BUILD_CMD, _VENDOR_CMD)],
)
def test_sealbot_refusal_reason_names_exactly_its_own_missing_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arm: str, must_contain: str,
    must_not_contain: str,
) -> None:
    """A reason that named both commands would be a checklist, not a diagnosis: the operator
    could not tell which step to run."""
    import mantis.bots.sealbot as sealbot_mod
    from mantis.bots.resolve import resolve_bot

    root = None if arm == "vendor_absent" else tmp_path
    monkeypatch.setattr(sealbot_mod, "find_vendor_root", lambda: root)
    with pytest.raises(RungUnresolvable) as exc:
        resolve_bot("sealbot", depth=5, opponent_sims=128)
    reason = exc.value.reason
    assert must_contain in reason, reason
    assert must_not_contain not in reason, reason
    assert "MANTIS_BOT_" not in reason, (
        f"a sealbot skip reason may not name the DELETED env channel: {reason}"
    )


# The depth-6 sealbot rung is EXCLUDED from the default battery.
def test_the_excluded_sealbot_depth_refuses_as_an_operator_authorized_skip() -> None:
    """The rung is minted in every config and cannot finish inside `eval.round_timeout_sec` at
    the measured 30.9 s/move, so it loud-skips instead of killing a round — in the SAME class the
    other ruled skips use, because the skip-class counter buckets on that marker and
    `build_absent` would read to an operator as a broken box."""
    from mantis.bots.resolve import _R326_EXCLUDED_SEALBOT_DEPTHS, resolve_bot

    depth = next(iter(_R326_EXCLUDED_SEALBOT_DEPTHS))
    with pytest.raises(RungUnresolvable) as exc:
        resolve_bot("sealbot", depth=depth, opponent_sims=None)
    reason = exc.value.reason
    assert "operator-authorized skip (R139)" in reason, reason
    assert _R326_EXCLUDED_SEALBOT_DEPTHS[depth] in reason, (
        "the grounds are the deliverable and must arrive VERBATIM, as kraken's and strix's do"
    )
    assert exc.value.rung == f"sealbot_d{depth}", (
        f"the refusal must name the RUNG, not the kind: an operator reading the log has to see "
        f"which sealbot rung was skipped, since another one is still live. got {exc.value.rung!r}"
    )


def test_the_exclusion_is_keyed_on_DEPTH_and_leaves_the_other_rungs_alone() -> None:
    """The exclusion is a statement about ONE depth, not about sealbot: `sealbot_d5` carries the
    gate's own sealbot signal, so catching the kind would silently disarm the gate's only
    resolvable opponent while looking like a narrow skip. Driven against the LIVE ladder."""
    import yaml

    from mantis.bots.resolve import _R326_EXCLUDED_SEALBOT_DEPTHS

    rungs = yaml.safe_load((_REPO / "configs" / "run6.yaml").read_text(encoding="utf-8"))
    sealbot_depths = {r["depth"] for r in rungs["eval"]["ladder"]["rungs"]
                      if r["bot"] == "sealbot"}
    excluded = set(_R326_EXCLUDED_SEALBOT_DEPTHS)
    assert sealbot_depths, "run5 mints no sealbot rung; this row would assert nothing"
    assert sealbot_depths - excluded, (
        f"the exclusion {sorted(excluded)} covers every sealbot depth run5 mints "
        f"{sorted(sealbot_depths)} — that disarms wr_sealbot, the gate's only resolvable "
        "opponent signal"
    )
    survivor = min(sealbot_depths - excluded)
    assert survivor not in _R326_EXCLUDED_SEALBOT_DEPTHS, survivor


def test_the_exclusion_fires_BEFORE_the_extension_probe() -> None:
    """An excluded rung reads the same in a warm checkout and a cold one: a guard sitting after
    the loader would give a broken-box diagnosis for a rung the operator ruled out. Driven by
    making the probe explode — the ruled reason must still come back."""
    import mantis.bots.resolve as resolve_mod

    depth = next(iter(resolve_mod._R326_EXCLUDED_SEALBOT_DEPTHS))
    original = resolve_mod._sealbot_mod.load_sealbot_modules
    try:
        def _explode() -> Any:
            raise AssertionError("the probe must not be reached for an excluded rung")
        resolve_mod._sealbot_mod.load_sealbot_modules = _explode
        with pytest.raises(RungUnresolvable) as exc:
            resolve_mod.resolve_bot("sealbot", depth=depth, opponent_sims=None)
    finally:
        resolve_mod._sealbot_mod.load_sealbot_modules = original
    assert "operator-authorized skip (R139)" in exc.value.reason, exc.value.reason


# The env channel is GONE — behaviour and source, two observers.
@pytest.mark.parametrize("kind", sorted(_DEAD_ENV_KEYS))
def test_setting_the_deleted_env_key_changes_nothing(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """The outcome CLASS is identical with the key deleted and with it set to a value shaped like
    the old contract's payload."""
    env_key = _DEAD_ENV_KEYS[kind]
    if kind == "sealbot":
        _no_vendor_root(monkeypatch)

    monkeypatch.delenv(env_key, raising=False)
    unset_outcome = _reason_or_resolved(kind, depth=5 if kind == "sealbot" else None)

    monkeypatch.setenv(env_key, "some_adapter_module:build")
    set_outcome = _reason_or_resolved(kind, depth=5 if kind == "sealbot" else None)

    assert set_outcome == unset_outcome, (
        f"{env_key} still steers resolution for {kind!r}: unset -> {unset_outcome!r}, "
        f"set -> {set_outcome!r}. DESIGN_A §2.2(2) deletes the channel; a key whose only "
        f"effect is to change which refusal string is printed is not a feature (R125/R79)."
    )


def test_no_mantis_bot_env_literal_survives_under_src() -> None:
    """A source scan for the deleted key, with its own detector self-test inline so the row
    cannot pass by scanning nothing."""
    needle = "MANTIS_BOT_"
    assert needle in "prefix MANTIS_BOT_SEALBOT suffix", "the detector itself must fire"

    offenders = [
        str(path.relative_to(_REPO))
        for path in sorted(_SRC.rglob("*.py"))
        if needle in path.read_text()
    ]
    assert offenders == [], (
        f"the deleted env channel survives under src/: {offenders}. An env key that can "
        f"point anywhere is a host-path surface wearing a disguise (DESIGN_A §2.2(2))."
    )


# No host path or default endpoint enters bots/ with the adapter.
def test_no_home_relative_or_url_literal_in_bots_sources() -> None:
    """The token classes the sibling census does not pin: `~`-leading paths and `http(s)://`
    endpoints, over AST string constants only, so prose about vendoring is not a false positive."""
    url_re = re.compile(r"https?://")
    assert url_re.search("see https://example.invalid/x") is not None, "detector must fire"

    offenders: list[str] = []
    for path in sorted(_BOTS_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            literal = node.value
            if literal.startswith("~"):
                offenders.append(f"{path.name}:{node.lineno} home-relative {literal!r}")
            elif url_re.search(literal) is not None:
                offenders.append(f"{path.name}:{node.lineno} endpoint {literal!r}")
    assert offenders == [], (
        f"bots/ carries a host path or a default endpoint: {offenders}. The vendor URL is "
        f"vendor/pins.toml's, and it is the only external string this phase permits."
    )
