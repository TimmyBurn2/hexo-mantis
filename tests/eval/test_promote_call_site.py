"""⊕ WP11-A DESIGN §a.4/§c.5 — the interface caution: WP-UNFREEZE will split `PromotionTarget`
(pool_hooks.py:64-76) into ActorSyncTarget/DeployTag. WP11-A may call the existing promotion
seam on gate-pass (zero-behavior parity with run3's flow) but must not deepen the coupling:
EXACTLY ONE gate-decision call site, marked with a seam comment referencing WP-UNFREEZE, and
no new code reading best_model/deploy state as a proxy for actor weights.

RED-at-import: `mantis.eval.promote` does not exist yet.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import mantis.eval.promote  # noqa: F401 — RED-at-import anchor: this module does not exist yet

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "mantis"

# pool.py / pool_hooks.py DEFINE the seam + forward their OWN self-calls to it; those are the
# pre-existing seam, excluded by path (dispatch: "pre-existing seam definitions excluded").
_EXCLUDED_PATHS = (_SRC / "selfplay" / "pool.py", _SRC / "selfplay" / "pool_hooks.py")

_TARGET_METHODS = ("sync_inference_weights", "update_checkpoint_step")

# O-G (FIX-PASS supplemental, DISPATCH_LOG-authorized): the census must ban ATTRIBUTE
# READS of best-model/anchor state as an actor-weight sync proxy, NOT the bare substring
# `best_model_step` — that substring is also the shipped `EvalPipelineLike` protocol's
# (coordinator/config.py:66-74) REQUIRED keyword parameter NAME, which DESIGN §c.3 orders
# `EvalPipeline.run_evaluation` to satisfy EXACTLY (pipeline.py's `best_model_step: "int |
# None"` parameter + its `best_model_step=best_model_step` pass-through are a protocol
# keyword name, never an attribute read). A genuine proxy read always has a leading `.`
# (`anchor_state.best_model_step`, `resolved_anchor.best_model`) — a bare parameter name,
# keyword argument, or local variable never does.
_PROXY_READ_RE = re.compile(r"anchor_state\.\w+|\.best_model(?:_step)?\b")


def _promotion_target_call_sites() -> list[tuple[Path, int, str]]:
    """Every `<expr>.sync_inference_weights(...)` / `<expr>.update_checkpoint_step(...)` CALL
    (attribute-method call, not a bare-name forwarder call, not a `def` definition) across
    src/mantis, excluding the pre-existing pool seam."""
    sites: list[tuple[Path, int, str]] = []
    for py_file in _SRC.rglob("*.py"):
        if py_file in _EXCLUDED_PATHS:
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _TARGET_METHODS
            ):
                sites.append((py_file, node.lineno, node.func.attr))
    return sites


def test_exactly_one_gate_decision_call_site() -> None:
    import mantis.eval.promote  # noqa: F401 — RED-at-import anchor; forces the module to exist

    sites = _promotion_target_call_sites()
    files = {p for p, _, _ in sites}
    assert files == {_SRC / "eval" / "promote.py"}, (
        f"sync_inference_weights/update_checkpoint_step must be called on a promotion target "
        f"from EXACTLY mantis/eval/promote.py; found call sites in: {sorted(str(f) for f in files)}"
    )
    methods_called = {m for _, _, m in sites}
    assert methods_called == set(_TARGET_METHODS), (
        f"promote.py must call BOTH {_TARGET_METHODS}, found only {methods_called}"
    )


def test_seam_comment_references_wp_unfreeze() -> None:
    promote_path = _SRC / "eval" / "promote.py"
    assert promote_path.is_file(), "mantis/eval/promote.py must exist"
    text = promote_path.read_text(encoding="utf-8")
    assert "WP-UNFREEZE" in text, (
        "the ONE gate-decision call site must carry the seam comment referencing WP-UNFREEZE "
        "(the PromotionTarget split coming next) — dispatch Interface caution"
    )


def test_no_new_actor_weight_proxy_reads() -> None:
    """mantis/eval + mantis/arena never READ `anchor_state.<attr>` / `<expr>.best_model` /
    `<expr>.best_model_step` to DECIDE a sync — only `promote.py` may reference them, and
    only to WRITE the post-decision update (recon T5 seam 7 is the known pre-existing
    coupling elsewhere; this WP must not deepen it with a second reader). A bare
    `best_model_step` occurrence with NO leading `.` (a parameter name, a keyword
    argument, a local variable) is NOT a proxy read — it is the shipped
    `EvalPipelineLike` protocol's required keyword parameter name (DESIGN §c.3: satisfy
    it EXACTLY), and is permitted (O-G)."""
    # RED-at-assertion anchor: without this, the census below would pass VACUOUSLY today —
    # mantis/eval already exists as a one-line skeleton (no promote.py yet), so an empty
    # census would find zero offenders for the wrong reason. Requiring promote.py to exist
    # makes this test fail today exactly like its two siblings above (R8/no-tautology law).
    assert (_SRC / "eval" / "promote.py").is_file(), (
        "mantis/eval/promote.py must exist — the census this test performs is meaningless "
        "(vacuously true) while the one file allowed to read these fields does not exist yet"
    )
    offenders: list[str] = []
    for pkg in ("eval", "arena"):
        pkg_dir = _SRC / pkg
        if not pkg_dir.is_dir():
            offenders.append(f"{pkg_dir} does not exist yet")
            continue
        for py_file in pkg_dir.rglob("*.py"):
            if py_file.name == "promote.py":
                continue
            text = py_file.read_text(encoding="utf-8")
            if _PROXY_READ_RE.search(text):
                offenders.append(str(py_file))
    assert offenders == [], (
        "no file outside mantis/eval/promote.py may READ anchor_state.*/best_model/"
        f"best_model_step as an actor-weight sync proxy: {offenders}"
    )
