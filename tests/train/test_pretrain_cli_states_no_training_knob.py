"""The pretrain CLI states no `train.*` value; the minted config does.

The CLI once carried code-side literal defaults for keys `TrainConfig` also mints, several
divergent from `configs/run6.yaml` (lr 2x, batch_size 2x, eta_min 50x), and
`BootstrapTrainer.__init__` carried its own `config.get(key, literal)` fallbacks — a fix a
`dict.get` would silently defeat, which is why the trainer is in scope here.

The rows read the parser OBJECT and the module's AST, never a hand-listed copy of the
shadowed names: `SHADOWED_TRAIN_KEYS` is imported from the CLI and `TrainConfig`'s own
fields are the other authority, so a later shadow lands red rather than passing a stale list.
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
_CONFIGS = sorted((_REPO / "configs").glob("*.yaml"))

#: The flag spellings the deleted keys had; the assertions derive their subject from
#: `SHADOWED_TRAIN_KEYS`, never from this map.
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


def test_every_shadowed_key_is_a_live_TrainConfig_leaf() -> None:
    """Prove every shadowed key is a live `TrainConfig` leaf, so the set below has a subject."""
    fields = set(TrainConfig.model_fields)
    missing = [k for k in SHADOWED_TRAIN_KEYS if k not in fields]
    assert missing == [], f"not TrainConfig leaves: {missing}"


def test_the_parser_carries_no_flag_for_any_shadowed_key() -> None:
    """Prove no shadowed key has a flag, read off the parser object by dest not by spelling."""
    dests = {a.dest for a in _build_arg_parser()._actions}
    readded = sorted(k for k in SHADOWED_TRAIN_KEYS if k in dests and k != "eta_min")
    assert readded == [], (
        f"{readded} is back on the argparse surface. F-816-25/R296(b): these values come from "
        f"the config and nowhere else — a flag beside a minted key is R79's duplicate authority."
    )


def test_the_surviving_eta_min_override_carries_no_literal_default() -> None:
    """Prove the surviving `--eta-min` override defaults to None, so the base comes from the config.

    An explicit operator override is not a shadow; its code-side `1e-5` against a minted
    0.0005 was what made it one.
    """
    (action,) = [a for a in _build_arg_parser()._actions if a.dest == "eta_min"]
    assert action.default is None, (
        f"--eta-min default is {action.default!r}; an override's absent value must mean "
        f"'use the config', never a second number."
    )


def test_config_is_REQUIRED_and_has_no_default_path() -> None:
    """Prove --config is required with no default path, so silence never picks the numbers."""
    (action,) = [a for a in _build_arg_parser()._actions if a.dest == "config"]
    assert action.required is True
    assert action.default is None

    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args(["--encoding", "v6"])


@pytest.mark.parametrize("flag", ["--lr", "--batch-size", "--aux-weight", "--weight-decay",
                                  "--aux-chain-weight"])
def test_each_deleted_flag_is_REJECTED_rather_than_ignored(flag: str) -> None:
    """Prove each deleted flag is rejected, never parsed and ignored under an old command line."""
    assert flag not in _parser_option_strings()
    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args(["--config", "x", "--encoding", "v6", flag, "1"])


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: p.name)
def test_training_terms_reproduces_the_YAML_own_numbers(path: Path) -> None:
    """Prove `training_terms` reproduces the YAML's own numbers, read from the file's text."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))["train"]
    terms = training_terms(load_config(path).train)

    assert terms["lr"] == pytest.approx(float(raw["lr"]))
    assert terms["weight_decay"] == pytest.approx(float(raw["weight_decay"]))
    assert terms["batch_size"] == int(raw["batch_size"])
    assert terms["pretrain_eta_min"] == pytest.approx(float(raw["eta_min"]))


def test_the_run5_divergences_the_row_measured_are_now_GONE() -> None:
    """Prove the measured divergences are gone as VALUES, not merely as an absent flag."""
    terms = training_terms(load_config(_REPO / "configs" / "run6.yaml").train)
    assert terms["lr"] == pytest.approx(0.001)          # was 0.002 on the parser
    assert terms["batch_size"] == 256                    # was 512
    assert terms["pretrain_eta_min"] == pytest.approx(0.0005)    # was 1e-5, the sixth shadow


def test_training_terms_is_the_ONLY_place_the_CLI_reads_these_off_a_config() -> None:
    """Prove by AST that every `train_cfg.<attr>` access in the CLI sits inside `training_terms`."""
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


def test_the_parser_guard_FIRES_against_a_parser_that_carries_a_shadow() -> None:
    """Prove the parser guard bites a planted shadow, so the rows above are not vacuous."""
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=0.002)
    dests = {a.dest for a in p._actions}
    assert [k for k in SHADOWED_TRAIN_KEYS if k in dests and k != "eta_min"] == ["lr"]


def test_the_stray_reader_guard_FIRES_against_a_second_reader() -> None:
    tree = ast.parse("def other():\n    return train_cfg.lr\n")
    stray = [
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "train_cfg"
    ]
    assert stray == ["lr"]
