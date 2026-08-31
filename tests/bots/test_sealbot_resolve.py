"""⊕ WP12-R Phase A / O-A1..O-A4 (DESIGN_A §2.2, PREREG_A §1) — the resolver rewrite.

At HEAD `bots/resolve.py` refuses all three external kinds through an ENV-KEY channel
(`resolve.py:35-39,61-71`), which DESIGN_A §2.2(2) deletes as argued (R125/R79): for
`sealbot` the authority for where the engine lives is `vendor/pins.toml` + `make vendor`,
and two authorities for one fact is R79's exact prohibition; for `kraken`/`strix` the key
is a silent-arming surface with nothing behind it, since R139 rules both out for run5 with
named grounds.

The defect each row is the ONLY witness to:

- **O-A1** — a sealbot rung that skips without saying which STEP is missing. Three
  environments (no vendor root / vendor root but no extension / the loader itself raised)
  must produce three PAIRWISE DISTINCT reasons, each naming the exact command that fixes
  it. A single "sealbot unavailable" string satisfies every other row in this file and is
  precisely what makes a box misconfiguration indistinguishable from a ruled skip.
- **O-A2** — a kraken/strix skip that reads like a dispatcher shortfall. R143 says these
  skips are OPERATOR-AUTHORIZED; R139's grounds are the words that say so, and they are
  PER RUNG. A paraphrase is a drift and cross-contamination is a false diagnosis.
- **O-A3** — the env channel surviving the rewrite. Two observers, deliberately: the
  BEHAVIOUR (set vs unset produce the identical outcome) and the SOURCE (no `MANTIS_BOT_`
  literal under `src/`). Either alone is defeatable — a dead key still reads as an arming
  surface to an operator, and a live key that happens to produce equal strings today is a
  host-path channel wearing a disguise.
- **O-A4** — a host path or default endpoint entering `bots/` with the adapter. Rule 7.
  Scoped to the tokens `tests/bots/test_protocol.py::test_no_host_path_tokens_in_bots_
  sources` does NOT cover (`~`-leading literals, `http(s)://`), so the two are one authority
  split by token class, never a duplicate pin.

SEAM (frozen here, ORACLE-FIRST — IMPL builds to it or files a grant):
  * `mantis.bots.sealbot.find_vendor_root() -> pathlib.Path | None` — walks up from
    `mantis.__file__` for a directory holding `vendor/pins.toml`; `None`, never a default
    path, when not found.
  * `mantis.bots.sealbot.load_sealbot_modules() -> tuple[Any, Any]` — `(minimax_module,
    game_module)`; raises `RungUnresolvable(rung="sealbot", reason=...)`.
  * `bots/resolve.py` reaches it THROUGH the module object (`_sealbot_mod.load_sealbot_
    modules()`), never a from-import binding — the SR-3 property, so `monkeypatch.setattr`
    on the module attribute is seen at call time.
  * `mantis.bots.resolve._R139_SKIP_GROUNDS: dict[str, str]` — DESIGN_A §2.2(3)'s mapping.

>300 justify: one resolver, one file. Every row here is an assertion about the SAME function's
refusal surface — which reason fires, whether two reasons can be confused for one another, and
whether the ordering that decides between them holds. Splitting them would put the ruled skips
(R139's kraken/strix, R326(e)'s excluded sealbot depth) in one file and the environment-state
refusals in another, and the whole point of the pairwise-distinctness rows is that a reader of
the log can tell those two classes apart: an oracle that can only see one class at a time
cannot assert they are distinguishable.
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

#: DESIGN_A §2.2(3): R139's own words, per rung. Asserted as EXACT substrings — a
#: paraphrase is a drift, and the phase's whole claim about kraken/strix is that a reader
#: of the log can tell a ruled skip from a broken one.
_R139_GROUNDS = {"kraken": "weights not cleanly accessible", "strix": "actively changing"}

#: The three env keys DESIGN_A §2.2(2) deletes. Named here, in the ORACLE, because after
#: the rewrite there is nowhere in `src/` left to read them from.
_DEAD_ENV_KEYS = {
    "sealbot": "MANTIS_BOT_SEALBOT",
    "kraken": "MANTIS_BOT_KRAKEN",
    "strix": "MANTIS_BOT_STRIX",
}

#: The two commands a skip reason must name. Not host paths and not endpoints: `make
#: vendor` is the repo's ONE vendoring mechanism (CLAUDE.md "Deliberately absent") and the
#: build invocation is DESIGN_A §2.6's, run inside the gitignored vendor tree.
_VENDOR_CMD = "make vendor"
_BUILD_CMD = "build_ext --inplace"

_RESOLVED = "<resolved>"


def _reason_or_resolved(kind: str, *, depth: int | None) -> str:
    """The outcome CLASS of one `resolve_bot` call, as a comparable string.

    Environment-robust on purpose: on a box where the extension is built, `sealbot`
    RESOLVES rather than raising, and an oracle that hard-required a raise would red for an
    environment reason rather than for its own. `RungUnresolvable` -> its `.reason`;
    a returned factory -> the `_RESOLVED` sentinel.
    """
    from mantis.bots.resolve import resolve_bot

    try:
        resolve_bot(kind, depth=depth, opponent_sims=128)
    except RungUnresolvable as exc:
        return exc.reason
    return _RESOLVED


def _no_vendor_root(monkeypatch: pytest.MonkeyPatch) -> None:
    import mantis.bots.sealbot as sealbot_mod

    monkeypatch.setattr(sealbot_mod, "find_vendor_root", lambda: None)


# ── O-A1: three environments, three distinct reasons, each naming its own missing step ──
def test_sealbot_refusal_reasons_are_pairwise_distinct_across_three_environments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """O-A1, all three arms plus the distinctness assertion, in ONE function.

    The one-function shape is PREREG_A §9 C-11's specified shape and it is what makes the
    `[unreached]` labels in M-A1 valid: under M-A1 (the loader stops raising) arm (a)'s
    `pytest.raises` fails at block exit and arms (b), (c) and the distinctness assertion
    below never execute. Post-conditions sit OUTSIDE every block (SR-6).
    """
    import mantis.bots.sealbot as sealbot_mod
    from mantis.bots.resolve import resolve_bot

    # (a) no vendor root at all -> name the fetch step.
    monkeypatch.setattr(sealbot_mod, "find_vendor_root", lambda: None)
    with pytest.raises(RungUnresolvable) as absent_exc:
        resolve_bot("sealbot", depth=5, opponent_sims=128)
    reason_no_vendor = absent_exc.value.reason
    assert absent_exc.value.rung == "sealbot"
    assert _VENDOR_CMD in reason_no_vendor, reason_no_vendor

    # (b) vendor root present, extension absent -> name the BUILD step. `tmp_path` holds no
    # `external/sealbot/current/minimax_cpp*.so`, which is the whole of the condition.
    monkeypatch.setattr(sealbot_mod, "find_vendor_root", lambda: tmp_path)
    with pytest.raises(RungUnresolvable) as unbuilt_exc:
        resolve_bot("sealbot", depth=5, opponent_sims=128)
    reason_no_build = unbuilt_exc.value.reason
    assert _BUILD_CMD in reason_no_build, reason_no_build

    # (c) the loader itself raised -> carry the underlying failure's repr. A loader that
    # swallowed it would report "not built" for an ABI mismatch, which is R145's exact
    # predicted failure wearing the wrong label.
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
    """O-A1's reason-SHAPE arms. A reason that named both commands would be a checklist,
    not a diagnosis: the operator could not tell which step to run."""
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


# ── O-A2: R139's grounds, verbatim and per rung ─────────────────────────────────────────
@pytest.mark.parametrize("kind", ["kraken", "strix"])
@pytest.mark.parametrize("direction", ["carries_own_grounds", "lacks_the_other_rungs"])
def test_operator_authorized_skip_carries_r139_grounds_per_rung(kind: str, direction: str) -> None:
    """O-A2. `direction` splits the two halves so a mutation that paraphrases ONE rung's
    grounds (M-A2) is attributable to that rung rather than to "the grounds test"."""
    from mantis.bots.resolve import resolve_bot

    other = "strix" if kind == "kraken" else "kraken"
    with pytest.raises(RungUnresolvable) as exc:
        resolve_bot(kind, depth=None, opponent_sims=128)
    reason = exc.value.reason
    if direction == "carries_own_grounds":
        assert _R139_GROUNDS[kind] in reason, reason
        assert "R139" in reason, reason
        assert "operator-authorized" in reason, (
            f"R143 calls these skips OPERATOR-AUTHORIZED, not a dispatcher shortfall; a "
            f"reader of the log must be able to tell a ruled skip from a broken one: {reason}"
        )
    else:
        assert _R139_GROUNDS[other] not in reason, (
            f"{kind}'s skip reason carries {other}'s grounds — the grounds are PER RUNG "
            f"(R139), and a shared string is a false diagnosis: {reason}"
        )


# ── R326(e): the depth-6 sealbot rung is EXCLUDED from the default battery ──────────────
def test_the_excluded_sealbot_depth_refuses_as_an_operator_authorized_skip() -> None:
    """R326(e). The rung is minted in all seven configs and cannot finish inside
    `eval.round_timeout_sec` at the measured 30.9 s/move, so it loud-skips per R139 instead of
    killing a round.

    The refusal must land in the SAME class kraken and strix use — `operator_authorized` —
    because the in-run skip-class counter (LAW-18/R164) buckets on that marker, and a ruled
    exclusion that reported as `build_absent` would read to an operator as a broken box.

    MUTATION THAT REDS IT: the exclusion removed, or its reason rewritten without the R139
    marker."""
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
    """The exclusion must be a statement about ONE depth, not about sealbot.

    `sealbot_d5` is the rung that carries `wr_sealbot` — the gate's own sealbot signal — so an
    exclusion that caught the kind rather than the depth would silently disarm the eval gate's
    only resolvable opponent while looking like a narrow skip. Driven against the LIVE ladder
    rather than a literal, so a re-minted ladder moves this row with it.

    MUTATION THAT REDS IT: the guard keyed on `kind == "sealbot"` instead of on the depth."""
    import yaml

    from mantis.bots.resolve import _R326_EXCLUDED_SEALBOT_DEPTHS

    rungs = yaml.safe_load((_REPO / "configs" / "run5.yaml").read_text(encoding="utf-8"))
    sealbot_depths = {r["depth"] for r in rungs["eval"]["ladder"]["rungs"]
                      if r["bot"] == "sealbot"}
    excluded = set(_R326_EXCLUDED_SEALBOT_DEPTHS)
    assert excluded < sealbot_depths, (
        f"the exclusion must be a PROPER subset of run5's sealbot depths {sorted(sealbot_depths)}; "
        f"excluding {sorted(excluded)} leaves nothing behind and disarms wr_sealbot"
    )
    survivor = min(sealbot_depths - excluded)
    assert survivor not in _R326_EXCLUDED_SEALBOT_DEPTHS, survivor


def test_the_exclusion_fires_BEFORE_the_extension_probe() -> None:
    """An excluded rung must read the same in a warm checkout and a cold one.

    If the guard sat after `load_sealbot_modules`, the reason on a box without the built
    extension would be `BUILD_ABSENT` — a broken-box diagnosis for a rung the operator ruled
    out — and the skip-class counter would bucket it as `build_absent`. Driven by making the
    probe explode: the ruled reason must still come back.

    MUTATION THAT REDS IT: the guard moved below the `try:`."""
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


def test_the_excluded_rungs_grounds_are_not_shared_with_kraken_or_strix() -> None:
    """The same per-rung discipline R139 imposes on kraken and strix, extended to the third
    ruled skip: a shared grounds string is a false diagnosis."""
    from mantis.bots.resolve import _R326_EXCLUDED_SEALBOT_DEPTHS

    for grounds in _R326_EXCLUDED_SEALBOT_DEPTHS.values():
        for kind, other in _R139_GROUNDS.items():
            assert other not in grounds, f"the depth exclusion carries {kind}'s grounds"


# ── O-A3: the env channel is GONE — behaviour and source, two observers ─────────────────
@pytest.mark.parametrize("kind", ["sealbot", "kraken", "strix"])
def test_setting_the_deleted_env_key_changes_nothing(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """O-A3 arm (a). Compares the outcome CLASS with the key deleted vs set to a value that
    looks like the old contract's payload.

    Only the sealbot parametrization pins the vendor environment (`_no_vendor_root`), and
    deliberately: kraken/strix must red HERE by ASSERTION on the two reasons differing —
    the defect this arm exists to catch — not by an `ImportError` on a seam they do not
    use. Under the shipped `resolve.py:63-71` the two reasons differ for all three kinds.
    """
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
    """O-A3 arm (b) / N-A5. A source scan, with its own detector self-test inline so the
    row cannot pass by scanning nothing (R81/R86)."""
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


# ── O-A4: no host path / default endpoint enters bots/ with the adapter ─────────────────
def test_no_home_relative_or_url_literal_in_bots_sources() -> None:
    """O-A4 (Rule 7). Token classes NOT covered by
    `tests/bots/test_protocol.py::test_no_host_path_tokens_in_bots_sources`, which pins
    `/home/` and `/`-leading absolute literals: this row adds `~`-leading paths and
    `http(s)://` endpoints. AST string constants only, so prose about vendoring in a
    docstring or a comment is not a false positive — and the vendor URL R139 requires lives
    in `vendor/pins.toml`, which is not under `src/`.
    """
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
