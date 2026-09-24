"""The pretrain CLI states no `train.*` value; the minted config does.

The CLI once carried code-side literal defaults for keys `TrainConfig` also mints, several
divergent from the shipped config (lr 2x, batch_size 2x, eta_min 50x). The rows read the parser
OBJECT and the module's AST against `TrainConfig`'s own fields, never a hand-listed copy, so a
later shadow lands red rather than passing a stale list.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pytest

from mantis.config import TrainConfig
from mantis.encoding import all_specs
from mantis.train.pretrain.cli import _build_arg_parser

_REPO = Path(__file__).resolve().parents[2]
_CLI = _REPO / "src" / "mantis" / "train" / "pretrain" / "cli.py"


def _parser_option_strings() -> set[str]:
    return {opt for action in _build_arg_parser()._actions for opt in action.option_strings}


def _shadows(parser: argparse.ArgumentParser) -> list[str]:
    """Parser dests that are `TrainConfig` leaves: each is a second authority beside a minted key."""
    return sorted({a.dest for a in parser._actions} & set(TrainConfig.model_fields))


def test_the_parser_carries_no_flag_for_any_TrainConfig_leaf() -> None:
    """Prove no `train.*` leaf has a flag, read off the parser object by dest not by spelling."""
    readded = _shadows(_build_arg_parser())
    assert readded == [], (
        f"{readded} is back on the argparse surface. These values come from the config and "
        "nowhere else — a flag beside a minted key is a duplicate authority."
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


@pytest.mark.parametrize("flag", ["--corpus-npz", "--no-compile"])
def test_a_parsed_but_unread_flag_is_REJECTED_rather_than_ignored(
    flag: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Prove the dead corpus/compile flags exit non-zero naming themselves, never parse and vanish."""
    graph = next(s.name for s in all_specs() if s.representation == "graph")
    argv = ["--config", "x", "--encoding", graph, flag] + (["x"] if flag == "--corpus-npz" else [])
    with pytest.raises(SystemExit) as exc:
        _build_arg_parser().parse_args(argv)
    assert exc.value.code != 0
    assert f"unrecognized arguments: {flag}" in capsys.readouterr().err


def test_every_parsed_flag_is_READ_by_the_cli() -> None:
    """Prove every parser dest is read as `args.<dest>`, so a parsed-and-ignored flag lands red."""
    tree = ast.parse(_CLI.read_text(encoding="utf-8"))
    read = {n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "args"}
    dests = {a.dest for a in _build_arg_parser()._actions if a.dest != "help"}
    assert sorted(dests - read) == [], "parsed but never read: the flag sets nothing"


def test_the_CLI_reads_no_train_term_off_the_config_itself() -> None:
    """Prove by AST that the CLI makes no `train_cfg.<attr>` read: the section goes whole to the resolvers."""
    tree = ast.parse(_CLI.read_text(encoding="utf-8"))
    stray = [
        f"{node.attr} at line {node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "train_cfg"
    ]
    assert stray == [], f"the CLI reads train terms itself: {stray}"


def test_the_parser_guard_FIRES_against_a_parser_that_carries_a_shadow() -> None:
    """Prove the parser guard bites a planted shadow, so the row above is not vacuous."""
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=0.002)
    assert _shadows(p) == ["lr"]


def test_the_stray_reader_guard_FIRES_against_a_second_reader() -> None:
    tree = ast.parse("def other():\n    return train_cfg.lr\n")
    stray = [
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "train_cfg"
    ]
    assert stray == ["lr"]
