"""⊕ WPUF Phase U ORACLE — O-U3 static half: the no-cross-read law, enforced structurally
(DESIGN_U §2.2 censuses S1–S5 + §2.4's unconditional-construction pin), each census with
its LAW-07 bite test (template: tests/selfplay/test_pool_surface.py H-12).

RED-at-import until IMPL lands `mantis.train.actor_sync` and `DeployTagHooks`.

PLACEMENT DEVIATIONS (logged in ORACLE_NOTES_U.md): DESIGN homes S4 in the rewritten
`tests/eval/test_promote_call_site.py` and E10's kept half (S5) in the edited
`tests/test_run_composition.py` — both existing files ORACLE-WRITE may not touch; both
censuses are therefore frozen HERE so the properties are oracle-guaranteed regardless of
how IMPL executes those on-list rewrites.

>300 justify (R8): five censuses + five bite tests + one AST pin share one walker/scanner
helper set; splitting the checkers from their bite tests would break the LAW-07 pairing.
"""
from __future__ import annotations

import ast
from pathlib import Path

import mantis.train.actor_sync  # noqa: F401 — RED-at-import anchor (module does not exist yet)
from mantis.eval.promote import DeployTagHooks

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "mantis"

_SYNC_METHODS = ("sync_inference_weights", "update_checkpoint_step")
# The pre-existing pool seam (definitions + self-forwarders), same exclusion as the
# WP11-A census (test_promote_call_site.py) — no new exclusions without adjudication.
_EXCLUDED = (_SRC / "selfplay" / "pool.py", _SRC / "selfplay" / "pool_hooks.py")

_S3_TOKENS = ("best_model", "anchor", "promot", "deploy", "gate", "eval_pipeline")
_S4_TOKENS = ("sync_inference_weights", "update_checkpoint_step", "promotion_target", "pool")
_S5_TOKENS = ("actor_lag", "actor_ckpt_step")

_DEPLOY_TAG_FIELDS = {
    "anchor_state", "best_model_path", "run_id", "encoding", "save_anchor", "guarded_load",
}


# ── census machinery (kept feedable so the bite tests can bite) ───────────────────────
def _sync_calls_in_source(text: str) -> list[str]:
    """Attribute-method CALLS of the two sync methods (never bare defs/forwarder names)."""
    found: list[str] = []
    for node in ast.walk(ast.parse(text)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in _SYNC_METHODS):
            found.append(node.func.attr)
    return found


def _files_calling_sync(root: Path, exclude: tuple[Path, ...] = ()) -> dict[Path, set[str]]:
    sites: dict[Path, set[str]] = {}
    for py_file in sorted(root.rglob("*.py")):
        if py_file in exclude:
            continue
        methods = _sync_calls_in_source(py_file.read_text(encoding="utf-8"))
        if methods:
            sites[py_file] = set(methods)
    return sites


def _tokens_present(text: str, tokens: tuple[str, ...]) -> list[str]:
    """Substring scan — deliberately blunt: catches lazy imports, duck-typed attribute
    reads and comments alike (the E10/`…even_lazily` shape)."""
    return sorted(t for t in tokens if t in text)


# ── S1: the sync call-site set is exactly {train/actor_sync.py}, calling BOTH ─────────
def test_sync_call_sites_are_exactly_the_actor_sync_engine() -> None:
    engine = _SRC / "train" / "actor_sync.py"
    sites = _files_calling_sync(_SRC, exclude=_EXCLUDED)
    assert set(sites) == {engine}, (
        "sync_inference_weights/update_checkpoint_step must be called from EXACTLY "
        f"mantis/train/actor_sync.py; found: {sorted(str(p) for p in sites)}"
    )
    assert sites[engine] == set(_SYNC_METHODS), (
        f"actor_sync.py must call BOTH {_SYNC_METHODS}; found {sites[engine]}"
    )


def test_s1_census_would_notice_a_second_sync_call_site() -> None:
    """LAW-07 bite: the walker flags a call and does NOT flag a bare definition."""
    assert _sync_calls_in_source("foo.sync_inference_weights(x)") == ["sync_inference_weights"]
    assert _sync_calls_in_source("def sync_inference_weights(self, sd): ...") == []


# ── S2: ZERO sync call sites in eval/arena (stated separately — E28's rationale) ──────
def test_no_sync_call_site_under_eval_or_arena() -> None:
    offenders: dict[Path, set[str]] = {}
    for pkg in ("eval", "arena"):
        offenders.update(_files_calling_sync(_SRC / pkg))
    assert offenders == {}, (
        f"the gate path must never re-acquire actor reach: {sorted(str(p) for p in offenders)}"
    )


def test_s2_census_would_notice_an_eval_side_call_site(tmp_path) -> None:
    bad = tmp_path / "eval" / "sneaky.py"
    bad.parent.mkdir()
    bad.write_text("def f(t):\n    t.update_checkpoint_step(3)\n")
    assert _files_calling_sync(tmp_path / "eval") == {bad: {"update_checkpoint_step"}}


# ── S3: the engine carries no deploy-side token, however reached ──────────────────────
def test_actor_sync_engine_carries_no_deploy_side_token() -> None:
    engine = _SRC / "train" / "actor_sync.py"
    assert engine.is_file(), "mantis/train/actor_sync.py must exist (the sync engine)"
    hits = _tokens_present(engine.read_text(encoding="utf-8"), _S3_TOKENS)
    assert hits == [], (
        f"actor_sync.py must contain none of the substrings {_S3_TOKENS}; found {hits} — "
        "any deploy→actor data path through the engine is banned, however reached"
    )


def test_s3_census_would_flag_a_deploy_token() -> None:
    assert _tokens_present("x = self.anchor_state", _S3_TOKENS) == ["anchor"]
    assert _tokens_present("plain sync code", _S3_TOKENS) == []


# ── S4: the deploy side has no attribute through which to reach a pool ────────────────
def test_deploy_tag_hooks_field_set_is_exactly_the_deploy_collaborators() -> None:
    assert set(DeployTagHooks.__dataclass_fields__) == _DEPLOY_TAG_FIELDS, (
        "DeployTagHooks must carry EXACTLY the deploy-side collaborators (R49 "
        f"unrepresentability pin); got {sorted(DeployTagHooks.__dataclass_fields__)}"
    )


def test_promote_module_carries_no_actor_surface_token() -> None:
    promote = _SRC / "eval" / "promote.py"
    hits = _tokens_present(promote.read_text(encoding="utf-8"), _S4_TOKENS)
    assert hits == [], (
        f"promote.py must contain none of the substrings {_S4_TOKENS}; found {hits}"
    )


def test_s4_census_would_flag_a_pool_reference() -> None:
    assert _tokens_present("hooks.promotion_target.sync_inference_weights(sd)", _S4_TOKENS) \
        == ["promotion_target", "sync_inference_weights"]
    assert "pool" in _tokens_present("target = pool", _S4_TOKENS)


# ── S5: eval never grows a lag/actor-step reader (the KEPT half of E10, GAPS-2) ───────
def test_no_actor_lag_mechanism_in_eval() -> None:
    offenders: list[str] = []
    for py_file in sorted((_SRC / "eval").rglob("*.py")):
        if _tokens_present(py_file.read_text(encoding="utf-8"), _S5_TOKENS):
            offenders.append(str(py_file.relative_to(_SRC)))
    assert offenders == [], (
        f"the lag mechanism is a train/watchdog property; eval may never read it: {offenders}"
    )


def test_s5_census_would_flag_an_eval_lag_reader() -> None:
    assert _tokens_present("lag = actor_ckpt_step_fn()", _S5_TOKENS) == ["actor_ckpt_step"]


# ── §2.4: compose_run builds ActorSync UNCONDITIONALLY (production wiring pin) ────────
def test_compose_run_builds_actor_sync_unconditionally() -> None:
    """`ActorSync(` is constructed in `compose_run`'s function body, under NO `if` —
    the same no-conditional shape as the pool.start() ordering pin. `actor_sync=None`
    on the coordinator is a unit-test affordance only; the ONE production wiring site
    must never make sync conditional on config/eval state (§2.4: not an R49 hole
    precisely because this pin exists)."""
    tree = ast.parse((_SRC / "run.py").read_text(encoding="utf-8"))
    compose = next(
        (n for n in tree.body
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "compose_run"),
        None,
    )
    assert compose is not None, "compose_run must exist in mantis/run.py"

    def _is_actor_sync_call(node: ast.AST) -> bool:
        return (isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", None)) == "ActorSync")

    calls = [n for n in ast.walk(compose) if _is_actor_sync_call(n)]
    assert calls, "compose_run must construct ActorSync (the continuous-sync engine)"

    conditional_nodes: set[int] = set()
    for branch in ast.walk(compose):
        if isinstance(branch, ast.If):
            for inner in ast.walk(branch):
                conditional_nodes.add(id(inner))
    guarded = [c for c in calls if id(c) in conditional_nodes]
    assert guarded == [], (
        "ActorSync construction sits under an `if` in compose_run — continuous sync "
        "must be unconditional (R49); a config/eval-gated construction is the old mode "
        "wearing a new guard"
    )
