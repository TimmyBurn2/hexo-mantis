"""⊕ WP-UNFREEZE (R50 rows E4/E5/E6) — the surviving half of the WP11-A call-site suite.

The split happened: `PromotionTarget` became `ActorSyncTarget` (actor seam) +
`DeployTagHooks` (deploy seam). The E4 call-site census relocated to
`tests/train/test_actor_sync_isolation.py` as S1/S2 (call sites == exactly
`{train/actor_sync.py}`; zero under eval/arena) and the promote-file pins live there as
S4 — the frozen copies are the ONLY copies (LAW-03: one census, one authority). The E5
seam-comment marker announced this split; executing the split discharged it.

What remains HERE is E6, KEPT VERBATIM: the anchor/best-model proxy-read ban across
`mantis/eval` + `mantis/arena`, with `promote.py` the sole exemption — one direction of
the no-cross-read law that predates the split and survives it unchanged.
"""
from __future__ import annotations

import re
from pathlib import Path

import mantis.eval.promote  # noqa: F401 — the one file allowed to touch these fields

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "mantis"

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
