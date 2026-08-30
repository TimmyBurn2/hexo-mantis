"""F-816-25 / R296(b) — the pretrain CLI states no `train.*` value; the minted config does.

THE DEFECT, MEASURED BEFORE THE FIX. `_build_arg_parser` carried code-side literal defaults for
five values `TrainConfig` also mints, three of them DIVERGENT at `configs/run5.yaml`:

    --lr             0.002   vs  train.lr                  0.001    (2x)
    --batch-size     512     vs  train.batch_size          256      (2x)
    --aux-weight     0.15    vs  train.aux_opp_reply_weight 0.0     (a head the config DISABLES)
    --weight-decay   0.0001  vs  train.weight_decay        0.0001   (agreed)
    --aux-chain-weight 0.0   vs  train.aux_chain_weight    0.0      (agreed)

**A SIXTH SHADOW AND FIVE MORE FALLBACKS, found by reading past the parser.** The row as filed
counted the argparse surface only. `_resume_into` carried `1e-5` for `eta_min` against
`train.eta_min: 0.0005` — a **50x** divergence and the largest of the set — and
`BootstrapTrainer.__init__` carried its OWN `config.get(key, literal)` fallback for `lr`,
`weight_decay`, `pretrain_total_steps` and `pretrain_eta_min`. Deleting the flags alone would
have left a fix a `dict.get` silently defeats, which is why the trainer is in this file's scope.

WHAT IS ASSERTED, AND WHY STRUCTURALLY. The rows below read the parser OBJECT and the module's
AST, never a hand-listed copy of the shadowed names: `SHADOWED_TRAIN_KEYS` is imported from the
CLI and `TrainConfig`'s own fields are the other authority, so a seventh shadow added later
lands RED here instead of passing a list nobody updated (R296(f), structure-not-text).

WHAT THIS FIX DOES NOT DO, stated so the row is not read as bigger than it is: it does not
unblock a BC-pretrain. That path is blocked by three larger things this file does not touch —
the certified corpus is axial MOVE LISTS with no encoder to training arrays in-tree, `data/`
has no producer (`save_corpus` has zero non-test callers), and this CLI is DENSE-ONLY while both
production configs are graph.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pytest
import yaml

from mantis.config import TrainConfig, load_config
from mantis.train.pretrain.cli import (
    SHADOWED_TRAIN_KEYS,
    _build_arg_parser,
    training_terms,
)

_REPO = Path(__file__).resolve().parents[2]
_CLI = _REPO / "src" / "mantis" / "train" / "pretrain" / "cli.py"
_TRAINER = _REPO / "src" / "mantis" / "train" / "pretrain" / "trainer.py"
_CONFIGS = sorted((_REPO / "configs").glob("*.yaml"))

#: The flag spellings the six keys had. Kept as the historical record of what was deleted —
#: the ASSERTION derives its subject from `SHADOWED_TRAIN_KEYS`, not from this map.
_DEAD_FLAGS: dict[str, str] = {
    "lr": "--lr",
    "weight_decay": "--weight-decay",
    "batch_size": "--batch-size",
    "aux_opp_reply_weight": "--aux-weight",
    "aux_chain_weight": "--aux-chain-weight",
    "eta_min": "--eta-min-DEAD",  # `--eta-min` SURVIVES as an override; its LITERAL is what died
}


def _parser_option_strings() -> set[str]:
    return {opt for action in _build_arg_parser()._actions for opt in action.option_strings}


# ── 1. every shadowed key is a real schema leaf, so the subject exists ──────────────────

def test_every_shadowed_key_is_a_live_TrainConfig_leaf() -> None:
    """The premise. A "shadow" of a key the schema does not mint would be no defect at all,
    and this row is what stops the set below from drifting into fiction."""
    fields = set(TrainConfig.model_fields)
    missing = [k for k in SHADOWED_TRAIN_KEYS if k not in fields]
    assert missing == [], f"not TrainConfig leaves: {missing}"


# ── 2. the parser states none of them ──────────────────────────────────────────────────

def test_the_parser_carries_no_flag_for_any_shadowed_key() -> None:
    """Read off the parser OBJECT, so a flag re-added under any spelling is caught by its
    dest rather than by a string search for the old name."""
    dests = {a.dest for a in _build_arg_parser()._actions}
    readded = sorted(k for k in SHADOWED_TRAIN_KEYS if k in dests and k != "eta_min")
    assert readded == [], (
        f"{readded} is back on the argparse surface. F-816-25/R296(b): these values come from "
        f"the config and nowhere else — a flag beside a minted key is R79's duplicate authority."
    )


def test_the_surviving_eta_min_override_carries_no_literal_default() -> None:
    """`--eta-min` SURVIVES, and the distinction is the point: an explicit operator override
    is not a shadow. What made it one was its code-side `1e-5` — 50x from `train.eta_min`'s
    minted 0.0005 — so the default must be `None` and the base must come from the config."""
    (action,) = [a for a in _build_arg_parser()._actions if a.dest == "eta_min"]
    assert action.default is None, (
        f"--eta-min default is {action.default!r}; an override's absent value must mean "
        f"'use the config', never a second number."
    )


def test_config_is_REQUIRED_and_has_no_default_path() -> None:
    """A default config path would be the same defect wearing a different hat: the CLI would
    once again decide the numbers when the operator said nothing."""
    (action,) = [a for a in _build_arg_parser()._actions if a.dest == "config"]
    assert action.required is True
    assert action.default is None

    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args(["--encoding", "v6"])


@pytest.mark.parametrize("flag", ["--lr", "--batch-size", "--aux-weight", "--weight-decay",
                                  "--aux-chain-weight"])
def test_each_deleted_flag_is_REJECTED_rather_than_ignored(flag: str) -> None:
    """A deleted flag that parsed silently would let an old command line run with the operator
    believing it took effect. argparse refuses an unknown option, and this row pins that it is
    still the behaviour after the deletion."""
    assert flag not in _parser_option_strings()
    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args(["--config", "x", "--encoding", "v6", flag, "1"])


# ── 3. the config is the authority, for every config the repo ships ─────────────────────

@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: p.name)
def test_training_terms_reproduces_the_YAML_own_numbers(path: Path) -> None:
    """Read the file's own text as the reference, not the loaded object, so this row would
    catch a resolver that transformed a value on its way through the schema."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))["train"]
    terms = training_terms(load_config(path).train)

    assert terms["lr"] == pytest.approx(float(raw["lr"]))
    assert terms["weight_decay"] == pytest.approx(float(raw["weight_decay"]))
    assert terms["batch_size"] == int(raw["batch_size"])
    assert terms["aux_opp_reply_weight"] == pytest.approx(float(raw["aux_opp_reply_weight"]))
    assert terms["aux_chain_weight"] == pytest.approx(float(raw["aux_chain_weight"]))
    assert terms["pretrain_eta_min"] == pytest.approx(float(raw["eta_min"]))


def test_the_run5_divergences_the_row_measured_are_now_GONE() -> None:
    """The three numbers F-816-25 actually measured, asserted as VALUES rather than as the
    absence of a flag — the difference between "the surface changed" and "the run would now
    use the minted number"."""
    terms = training_terms(load_config(_REPO / "configs" / "run5.yaml").train)
    assert terms["lr"] == pytest.approx(0.001)          # was 0.002 on the parser
    assert terms["batch_size"] == 256                    # was 512
    assert terms["aux_opp_reply_weight"] == pytest.approx(0.0)   # was 0.15
    assert terms["pretrain_eta_min"] == pytest.approx(0.0005)    # was 1e-5, the sixth shadow


def test_training_terms_is_the_ONLY_place_the_CLI_reads_these_off_a_config() -> None:
    """AST, not grep: every `train_cfg.<attr>` access in the CLI must sit inside
    `training_terms`. A second reader elsewhere would be a second authority again, which is
    exactly the shape the fix removed."""
    tree = ast.parse(_CLI.read_text(encoding="utf-8"))
    inside = {
        lineno
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef) and fn.name == "training_terms"
        for node in ast.walk(fn)
        if (lineno := getattr(node, "lineno", None)) is not None
    }
    stray = [
        f"{node.attr} at line {node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "train_cfg"
        and node.lineno not in inside
    ]
    assert stray == [], f"train_cfg read outside training_terms: {stray}"


# ── 4. the trainer no longer substitutes a number for a missing one ─────────────────────

@pytest.mark.parametrize("key", ["lr", "weight_decay", "pretrain_total_steps", "pretrain_eta_min"])
def test_the_trainer_reads_its_terms_by_SUBSCRIPT_not_by_get_with_a_default(key: str) -> None:
    """The half a flag deletion cannot reach. `BootstrapTrainer` carried
    `config.get("lr", 0.002)` and three siblings, so a config-only CLI whose dict lost a key
    would have been silently topped up with the very literal the fix removed.

    Asserted on the AST rather than by constructing a trainer, because constructing one needs
    a torch model and a device and this claim is about the source, not about a run.
    """
    tree = ast.parse(_TRAINER.read_text(encoding="utf-8"))
    offenders = [
        f"line {node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "config"
        and len(node.args) == 2
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == key
    ]
    assert offenders == [], (
        f"BootstrapTrainer still defaults {key!r} at {offenders}. A `.get` fallback is a "
        f"code-side default (R1) and re-opens F-816-25 one layer below the CLI."
    )


# ── 5. LAW-07: the guards are shown able to fire ───────────────────────────────────────

def test_the_parser_guard_FIRES_against_a_parser_that_carries_a_shadow() -> None:
    """Without this the parser rows would pass vacuously the day someone renamed a dest."""
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=0.002)
    dests = {a.dest for a in p._actions}
    assert [k for k in SHADOWED_TRAIN_KEYS if k in dests and k != "eta_min"] == ["lr"]


def test_the_trainer_guard_FIRES_against_a_live_two_arg_get() -> None:
    """Drive the AST predicate against source that DOES carry the pattern, so a green run
    means "the pattern is absent", never "the matcher stopped matching"."""
    tree = ast.parse('lr = float(config.get("lr", 0.002))\n')
    hits = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "config"
        and len(node.args) == 2
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "lr"
    ]
    assert len(hits) == 1


def test_the_stray_reader_guard_FIRES_against_a_second_reader() -> None:
    tree = ast.parse("def other():\n    return train_cfg.lr\n")
    stray = [
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "train_cfg"
    ]
    assert stray == ["lr"]
