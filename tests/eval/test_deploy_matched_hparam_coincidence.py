"""The eval path's code-side literals EQUAL run5's config values.

Thirteen hyper-parameters on the LAW-15 deploy-matched eval path are code-side literals measured
equal to `configs/run6.yaml` by coincidence of defaults, not by threading, and nothing detected
the day they stopped being equal. This file is that detector: it PINS THE COINCIDENCE rather
than threading the config (a behaviour change on a frozen parity surface), so retuning run5
without following the eval path names the decoupling instead of letting the promotion bar stop
measuring the net that ships. Both literal sets are read FROM THEIR SOURCE — the pyo3 signature,
the inline dict — never transcribed.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_RUN5 = _REPO / "configs" / "run6.yaml"
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
    """The `{"inference": {...}}` dict node that `LocalInferenceEngine.__init__` hands to the
    graph `InferenceServer`."""
    tree = ast.parse(_INFERENCE_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
        if "inference" in keys:
            return node
    pytest.fail("could not locate the inline inference dict literal")


def _split_literal_and_threaded(section: str) -> tuple[dict[str, Any], set[str]]:
    """`(constant entries, keys whose value is NOT a constant)` for one section — an
    `ast.Constant` value is a code-side literal that can drift from the config, any other
    expression is READ from something, and a key threaded later moves sides on its own."""
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


# D-15: `DeployHeadPlayer.new_game` builds `MCTSTree()` on the pyo3 ctor defaults while
# self-play threads the same knobs from config.
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


# D-23: the eval engine's InferenceServer knobs. The six mirrored `inference.*` keys went with
# the dense path, so the coverage row below is what keeps a NEW literal from appearing unwatched.
_INFERENCE_KEYS: list[str] = []

#: Keys run5 declares that the inline literal DELIBERATELY does not mirror; a CLOSED set of one,
#: since a second unmirrored key reds the coverage row. `inference.fused_graph_caps` is threaded
#: in as a resolver-produced frozen dataclass, so a cap hardcoded here would be a SECOND
#: AUTHORITY over one byte budget; an AST census in tests/selfplay makes writing one impossible.
_DELIBERATELY_NOT_IN_THE_LITERAL = {"fused_graph_caps"}

#: Keys in the inline dict that are THREADED rather than written, so there is no literal to
#: drift; the coverage row proves the move STRUCTURALLY, not on this comment. Un-threaded,
#: `inference_batch_size`/`inference_max_wait_ms` cost 1.76 of the eval path's 5.30 ms/sim
#: (33 %) at the single-stream deploy head: supply 8 against a collector threshold of 32.
_THREADED_NOT_LITERAL = {"inference_batch_size", "inference_max_wait_ms"}


def test_the_literal_covers_every_key_run5_declares() -> None:
    """Coverage, not just agreement: a NEW key in run5's inference block that the literal omits
    is the same defect one level out, and parametrising over a stale list would hide it. The
    exception set is CLOSED — a key is excused only by a ruling that says the literal must not
    carry it."""
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


def test_the_train_SECTION_IS_GONE_and_law06_still_pins_the_dtype() -> None:
    """The inline server dict must not grow a `train` section back, and LAW-06's graph pin must
    still hold.

    The dict once wrote `train.amp_dtype = "bf16"` as a literal while run5 declared `fp16`; the
    difference was inert only because `amp_dtype_for` pins the graph path to bfloat16 regardless.
    The value was then threaded and the key deleted, so what remains to pin is that neither the
    section nor the config row comes back.
    """
    import torch

    from mantis.model.amp import amp_dtype_for

    outer = _inline_inference_dict()
    sections = [k.value for k in outer.keys if isinstance(k, ast.Constant)]
    assert "train" not in sections, (
        "the inline server dict in inference_local.py grew a `train` section again. The only "
        "thing it ever carried was `amp_dtype`, which R346(f) deleted; a re-added section is a "
        "second dtype authority on the one construction path with no config to be the first."
    )
    assert "amp_dtype" not in _run5()["train"], (
        "`train.amp_dtype` is back in run5. LAW-06 pins the graph autocast dtype in code; a "
        "config row for it is the second authority this deletion removed."
    )
    assert amp_dtype_for("graph") is torch.bfloat16, (
        "LAW-06's graph pin is what made the OLD literal-vs-declared divergence inert. The "
        "divergence is gone, but the pin is still what the graph path's dtype rests on."
    )
