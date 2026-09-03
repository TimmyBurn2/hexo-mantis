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


def _inline_inference_dict() -> ast.Dict:
    """The `{"inference": {...}, "train": {...}}` dict node that
    `LocalInferenceEngine.__init__` hands to the graph `InferenceServer`."""
    tree = ast.parse(_INFERENCE_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
        if "inference" in keys and "train" in keys:
            return node
    pytest.fail("could not locate the inline inference dict literal")


def _split_literal_and_threaded(section: str) -> tuple[dict[str, Any], set[str]]:
    """`(constant entries, keys whose value is NOT a constant)` for one section.

    The split IS the distinction this file polices. A key whose value is an `ast.Constant`
    is a code-side literal and can drift from the config; a key whose value is any other
    expression is READ from something, and there is no coincidence left to pin. Deriving the
    two sets structurally means a key that gets threaded later moves sides on its own rather
    than needing this file edited to stay honest.
    """
    outer = _inline_inference_dict()
    for key_node, val_node in zip(outer.keys, outer.values):
        if not (isinstance(key_node, ast.Constant) and key_node.value == section):
            continue
        assert isinstance(val_node, ast.Dict), f"the {section!r} entry is not a dict literal"
        literals: dict[str, Any] = {}
        threaded: set[str] = set()
        for k, v in zip(val_node.keys, val_node.values):
            assert isinstance(k, ast.Constant), f"a non-constant key in the {section!r} dict"
            if isinstance(v, ast.Constant):
                literals[k.value] = v.value
            else:
                threaded.add(k.value)
        return literals, threaded
    pytest.fail(f"the inline dict has no {section!r} section")


def _inline_inference_literal() -> dict[str, dict[str, Any]]:
    """The CONSTANT entries of both sections, keyed as before."""
    return {
        "inference": _split_literal_and_threaded("inference")[0],
        "train": _split_literal_and_threaded("train")[0],
    }


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
    "trace_inference", "compile_inference", "compile_inference_mode",
    "compile_inference_dynamic", "perf_timing", "perf_sync_cuda",
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

#: Keys that ARE in the inline dict but are THREADED rather than written — so there is no
#: literal to drift and nothing for the coincidence rows above to pin. Each carries the
#: ruling that moved it here, and `test_the_literal_covers_every_key_run5_declares` proves
#: the move STRUCTURALLY (the value is not an `ast.Constant`) rather than on this comment.
#:
#: `inference_batch_size` / `inference_max_wait_ms` (PERF-TRANCHE-1 G-2, ledger F-2). These
#: were the two rows this file was written to police, and policing them was always the
#: second-best option the queue offered — it detects the drift, it does not prevent it. The
#: ledger measured what the un-threaded pair costs on the arm LAW-15 reads a promotion bar
#: off: at the single-stream deploy head, supply 8 against a collector threshold the literal
#: set to 32, **1.76 of the eval path's 5.30 ms/sim — 33 %** — is the collector's own
#: deadline. They are now resolved in the parent through
#: `mantis.config.resolve.inference_batching` and carried across the process seam on
#: `RoundSpec`, exactly as `fused_graph_caps` is.
_THREADED_NOT_LITERAL = {"inference_batch_size", "inference_max_wait_ms"}


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
    sets = [set(_INFERENCE_KEYS), _DELIBERATELY_NOT_IN_THE_LITERAL, _THREADED_NOT_LITERAL]
    assert set().union(*sets) == set(_run5()["inference"]), (
        "run5's inference block and this oracle's key lists have diverged"
    )
    for i, a in enumerate(sets):
        for b in sets[i + 1:]:
            assert not (a & b), (
                "a key is listed in two of {mirrored, deliberately-absent, threaded}; the "
                "three lists must PARTITION run5's inference block, or the coverage claim "
                "above is vacuous"
            )
    literals, threaded = _split_literal_and_threaded("inference")
    for key in _DELIBERATELY_NOT_IN_THE_LITERAL:
        assert key not in literals and key not in threaded, (
            f"{key!r} is excused from the coincidence BECAUSE the literal must not carry it "
            f"(F-816-10 D-1) — and it now does. Either the ruling changed or a second "
            f"authority over one byte budget was just written into inference_local.py."
        )
    for key in _THREADED_NOT_LITERAL:
        assert key in threaded, (
            f"{key!r} is recorded as THREADED (G-2) but the inline dict now writes it as a "
            f"constant. That is the hardcode the threading removed, put back: the eval "
            f"path's collector geometry would again be a number nobody minted, on the one "
            f"arm LAW-15 reads a promotion bar off (ledger F-2)."
        )


def test_the_amp_dtype_LITERAL_IS_GONE_and_the_declared_value_is_threaded() -> None:
    """THE SHARP HALF — and the shape of this row CHANGED at AUDIT-1 F-31, which is worth
    saying rather than quietly rewriting.

    IT USED TO ASSERT A DISAGREEMENT. `inference_local.py`'s inline server dict wrote
    `train.amp_dtype = "bf16"` as a LITERAL while run5 DECLARES `fp16`; the two differed, and
    the difference was inert only because `amp_dtype_for` pins the graph path to bfloat16
    regardless (LAW-06). This test asserted both facts so that a relaxed pin would make the
    divergence visible — a good instrument for a defect that was being lived with.

    F-31 REMOVED THE DEFECT, so the instrument changes with it: the literal is THREADED now
    (`"amp_dtype": amp_dtype`, from the caller's declared value), and there is no disagreement
    left to watch. What is pinned instead is that the literal does not come BACK, and that
    LAW-06's pin — which is why the old divergence was survivable — still holds.
    """
    import torch

    from mantis.model.amp import amp_dtype_for

    literals, threaded = _split_literal_and_threaded("train")
    assert "amp_dtype" not in literals, (
        "`train.amp_dtype` is a LITERAL in inference_local.py's inline server dict again. "
        "That is a second dtype authority on the one construction path with no config to be "
        "the first, and it is what AUDIT-1 F-31 removed."
    )
    assert "amp_dtype" in threaded, (
        "`train.amp_dtype` is neither a literal nor threaded in the inline dict — the key "
        "vanished. The server hard-reads it (R30b, no fallback), so a missing key is a boot "
        "failure on the graph eval path, not a simplification."
    )
    assert amp_dtype_for("graph", _run5()["train"]["amp_dtype"]) is torch.bfloat16, (
        "LAW-06's graph pin is what made the OLD literal-vs-declared divergence inert. The "
        "divergence is gone, but the pin is still what the graph path's dtype rests on."
    )
