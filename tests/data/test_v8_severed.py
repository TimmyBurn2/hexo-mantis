"""O4e — v8-severed census on the replay dispatcher.

Confirms the KILL: ``mantis/data/replay.py`` carries no ``dataset_v8`` import and no
``v8`` / ``v8_canvas_realness`` dispatch key; the supported set is exactly the three
registered corpus encodings.
"""
from __future__ import annotations

import ast
from pathlib import Path

from mantis.data import replay as replay_mod
from mantis.data.replay import _SUPPORTED

_REPLAY_SRC = Path(replay_mod.__file__).read_text()


def test_supported_set_has_no_v8() -> None:
    assert _SUPPORTED == ("v6", "v6w25", "v6_live2_ls")
    assert "v8" not in _SUPPORTED
    assert "v8_canvas_realness" not in _SUPPORTED


def test_no_dataset_v8_import() -> None:
    tree = ast.parse(_REPLAY_SRC)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported += [a.name for a in node.names]
    assert not any("dataset_v8" in name or "v8" in name for name in imported)


def test_no_v8_dispatch_string_literals() -> None:
    tree = ast.parse(_REPLAY_SRC)
    literals = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "v8" not in literals
    assert "v8_canvas_realness" not in literals
