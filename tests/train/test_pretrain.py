"""O-PRE — pretrain dataset conformance (WP10 §a.7).

Gates: `train/pretrain/dataset.py` has NO `dataset_v8` import and NO `v8` / `v8_canvas_realness`
dispatch key (v8 never crosses — the registry has no v8); the dataset imports the WP9-relocated
`mantis.data` replayers + augment (train → data is DAG-legal). Bites: v8 crossing.
"""
from __future__ import annotations

import ast
from pathlib import Path

import mantis.train.pretrain.dataset as dataset

_DATASET_SRC = Path(dataset.__file__).read_text()


def _imported_modules(src: str) -> set[str]:
    """Every module referenced by an `import`/`from … import` in the source (AST-exact — not a
    substring scan)."""
    mods: set[str] = set()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


# ── v8 is SEVERED ─────────────────────────────────────────────────────────────────────────
def test_dataset_no_dataset_v8_import() -> None:
    """No `dataset_v8` import (the KILLED v8 dataset module) — anywhere in the source."""
    assert "dataset_v8" not in _DATASET_SRC
    assert not any("dataset_v8" in m for m in _imported_modules(_DATASET_SRC))


def test_dataset_no_v8_dispatch_key() -> None:
    """No `v8` / `v8_canvas_realness` dispatch KEY — no `"v8"` string literal drives a branch
    (the v8 augment branch is severed). Prose in the docstring is fine; a v8 STRING LITERAL is
    what a dispatch key would be, and there must be none."""
    assert "v8_canvas_realness" not in _DATASET_SRC
    tree = ast.parse(_DATASET_SRC)
    v8_literals = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value == "v8"
    ]
    assert not v8_literals, "dataset.py must carry no 'v8' string-literal dispatch key"


# ── imports the WP9 mantis.data replayers + augment ───────────────────────────────────────
def test_dataset_imports_wp9_data_seams() -> None:
    """The dataset imports the WP9-relocated `mantis.data.replay` replayer + the
    `mantis.data.augment` policy-scatter helper (train → data DAG-legal)."""
    mods = _imported_modules(_DATASET_SRC)
    assert "mantis.data.replay" in mods, "must import the WP9 mantis.data.replay replayer"
    assert "mantis.data.augment" in mods, "must import mantis.data.augment.get_policy_scatters"
    # functional: the re-exported replayer + the collate/dataset are importable.
    assert callable(dataset.replay_game_to_triples)
    assert callable(dataset.make_augmented_collate)
    assert dataset.AugmentedBootstrapDataset is not None


def test_dataset_public_surface() -> None:
    """The pretrain package re-exports the dataset surface (no v8 symbols)."""
    import mantis.train.pretrain as pretrain

    for name in ("AugmentedBootstrapDataset", "make_augmented_collate", "BootstrapTrainer", "validate"):
        assert hasattr(pretrain, name)
    assert not any("v8" in n for n in getattr(pretrain, "__all__", []))
