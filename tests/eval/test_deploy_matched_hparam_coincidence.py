"""ADJ-WP12R-8 producer: the eval path's code-side literals EQUAL run5's config values.

Census sites D-15 and D-23 found thirteen hyper-parameters on the LAW-15 deploy-matched
eval path that are code-side literals rather than threaded config (R1 forbids code-side
defaults). They were measured EQUAL to `configs/run5.yaml` — but equal *by coincidence of
defaults, not by threading*. Nothing detects the day they stop being equal.

This file is the detector. It does not thread the config (that is a design decision the
queue routes to the architect, and threading is a behaviour change on a frozen parity
surface); it PINS THE COINCIDENCE, which is the queue row's own second option. If anyone
retunes run5 without following the eval path, these reds and names the decoupling instead
of letting the promotion bar silently stop measuring the net that ships.

Both literal sets are read FROM THEIR SOURCE (the pyo3 signature; the inline dict), never
transcribed here — a transcribed copy would drift with the thing it claims to pin.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_RUN5 = _REPO / "configs" / "run5.yaml"
_MCTS_RS = _REPO / "crates" / "mantis-bridge" / "src" / "mcts.rs"
_INFERENCE_PY = _REPO / "src" / "mantis" / "selfplay" / "inference_local.py"


def _run5() -> dict[str, Any]:
    return yaml.safe_load(_RUN5.read_text())


def _pyo3_mctstree_defaults() -> dict[str, Any]:
    """Parse `#[pyo3(signature = (...))]` on `PyMCTSTree::new` — the defaults
    `DeployHeadPlayer.new_game` gets by calling `MCTSTree()` with no arguments."""
    source = _MCTS_RS.read_text()
    match = re.search(
        r"#\[pyo3\(signature = \((c_puct[^)]*)\)\)\]", source, re.DOTALL
    )
    assert match, "could not locate the PyMCTSTree::new pyo3 signature"
    out: dict[str, Any] = {}
    for part in match.group(1).split(","):
        if "=" not in part:
            continue
        key, raw = (piece.strip() for piece in part.split("=", 1))
        out[key] = {"true": True, "false": False}.get(raw, raw)
    return out


def _inline_inference_literal() -> dict[str, Any]:
    """Extract the `{"inference": {...}, "train": {...}}` dict literal that
    `LocalInferenceEngine.__init__` hands to the graph `InferenceServer`."""
    tree = ast.parse(_INFERENCE_PY.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
        if "inference" in keys and "train" in keys:
            return ast.literal_eval(node)
    pytest.fail("could not locate the inline inference dict literal")


# ── D-15: the deploy head's MCTS hyper-parameters ────────────────────────────────────
# `DeployHeadPlayer.new_game` builds `MCTSTree()` on the pyo3 ctor defaults while
# self-play threads these from config (`runner/game.rs:228-231`).
@pytest.mark.parametrize(
    "ctor_key, config_key",
    [
        ("c_puct", "c_puct"),
        ("fpu_reduction", "fpu_reduction"),
        ("quiescence_enabled", "quiescence_enabled"),
        ("quiescence_blend_2", "quiescence_blend_2"),
    ],
)
def test_deploy_head_mcts_default_equals_run5(ctor_key: str, config_key: str) -> None:
    ctor = _pyo3_mctstree_defaults()[ctor_key]
    configured = _run5()["selfplay"]["mcts"][config_key]
    if isinstance(configured, bool):
        assert ctor is configured, (
            f"MCTSTree ctor default {ctor_key}={ctor!r} no longer equals run5's "
            f"selfplay.mcts.{config_key}={configured!r}. The eval deploy head does NOT "
            f"read the config — it would keep the ctor default while self-play moved, so "
            f"the promotion bar would stop being deploy-matched (LAW-15). Thread the "
            f"value or re-rule ADJ-WP12R-8."
        )
    else:
        assert float(ctor) == float(configured), (
            f"MCTSTree ctor default {ctor_key}={ctor!r} no longer equals run5's "
            f"selfplay.mcts.{config_key}={configured!r} — see ADJ-WP12R-8."
        )


# ── D-23: the eval engine's InferenceServer hyper-parameters ─────────────────────────
_INFERENCE_KEYS = [
    "inference_batch_size", "inference_max_wait_ms", "trace_inference",
    "compile_inference", "compile_inference_mode", "compile_inference_dynamic",
    "perf_timing", "perf_sync_cuda",
]

#: Keys run5 declares that the inline literal DELIBERATELY does not mirror, each with the
#: ruling that says so. A CLOSED set of one: a second unmirrored key still reds the coverage
#: row below, so this is an exception with a name and not a relaxation.
#:
#: `inference.fused_graph_caps` (F-816-10 D-1) is the graph inference forward's memory bound,
#: and it is THREADED into `LocalInferenceEngine` as a resolver-produced frozen dataclass
#: rather than written into the literal. That is the whole content of the ruling: a cap
#: hardcoded here would be a SECOND AUTHORITY over one byte budget — the exact defect
#: `MicrobatchCapsConfig`'s docstring refuses — sitting on the ONE construction path with no
#: config to be the first, on the arm that runs in the eval child with its own allocator. So
#: the coincidence this file polices does not apply: there is no literal to drift from the
#: config, because the value ARRIVES from the config. `tests/selfplay/
#: test_fused_graph_caps_construction.py::test_fg6_07_no_cap_value_is_hardcoded_at_the_
#: standalone_construction_site` is the AST census that makes writing one here impossible,
#: and it is the row that would red if someone "fixed" this exception by adding the key.
_DELIBERATELY_NOT_IN_THE_LITERAL = {"fused_graph_caps"}


@pytest.mark.parametrize("key", _INFERENCE_KEYS)
def test_inline_inference_literal_equals_run5(key: str) -> None:
    literal = _inline_inference_literal()["inference"][key]
    configured = _run5()["inference"][key]
    assert literal == configured, (
        f"inference_local.py's inline literal {key}={literal!r} no longer equals run5's "
        f"inference.{key}={configured!r}. The eval engine builds its graph InferenceServer "
        f"from that literal, not from the config — see ADJ-WP12R-8 (census site D-23)."
    )


def test_the_literal_covers_every_key_run5_declares() -> None:
    """Coverage, not just agreement: a NEW key in run5's inference block that the literal
    omits is the same defect one level out, and parametrising over a stale list would hide
    it.

    The exception set is CLOSED and each member carries its ruling: a key is excused from the
    coincidence only by a ruling that says the literal must NOT carry it, never by a key
    quietly appearing on one side."""
    assert set(_INFERENCE_KEYS) | _DELIBERATELY_NOT_IN_THE_LITERAL == set(
        _run5()["inference"]
    ), "run5's inference block and this oracle's key list have diverged"
    assert not (set(_INFERENCE_KEYS) & _DELIBERATELY_NOT_IN_THE_LITERAL), (
        "a key is listed as BOTH mirrored and deliberately-absent; the two lists must "
        "partition run5's inference block, or the coverage claim above is vacuous"
    )
    literal = _inline_inference_literal()["inference"]
    for key in _DELIBERATELY_NOT_IN_THE_LITERAL:
        assert key not in literal, (
            f"{key!r} is excused from the coincidence BECAUSE the literal must not carry it "
            f"(F-816-10 D-1) — and it now does. Either the ruling changed or a second "
            f"authority over one byte budget was just written into inference_local.py."
        )


def test_amp_dtype_disagrees_and_is_erased_only_by_the_law06_pin() -> None:
    """THE SHARP HALF, asserted as what it is rather than papered over.

    The literal hands `train.amp_dtype = "bf16"`; run5 DECLARES `fp16`. They differ. The
    difference is inert only because `amp_dtype_for` pins the graph path to bfloat16
    regardless (LAW-06). This test asserts BOTH facts, so that if LAW-06's pin is ever
    relaxed the disagreement becomes visible instead of silently taking effect.
    """
    from mantis.model.amp import amp_dtype_for
    import torch

    literal = _inline_inference_literal()["train"]["amp_dtype"]
    declared = _run5()["train"]["amp_dtype"]
    assert literal == "bf16" and declared == "fp16", (
        f"the known disagreement changed shape: literal={literal!r} declared={declared!r}"
    )
    assert amp_dtype_for("graph", declared) is torch.bfloat16, (
        "LAW-06's graph pin is what makes the amp_dtype disagreement inert; it no longer "
        "holds, so the eval path's bf16 literal now genuinely diverges from run5's fp16"
    )
