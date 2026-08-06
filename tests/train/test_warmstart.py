"""O-WARM — warm-start seam conformance (WP10 §a.6/§c.5).

Gates: `head_dir` is a REQUIRED explicit parameter; a source census finds NO `/home/`, no
`/Users/`, no `expanduser` in `train/warmstart.py` (the KILLed `_HEADSWAP_AB_DIR` personal-path
default); a missing `head_dir` raises loudly. Bites: reintroducing a personal-path default
(host-coupling ban / CLAUDE.md R1).
"""
from __future__ import annotations

from pathlib import Path

import pytest

import mantis.train.warmstart as warmstart
from mantis.train.warmstart import (
    HEAD_FILE_BY_TYPE,
    default_head_for_arm,
    maybe_warmstart_value_head,
    resolve_warmstart_head_file,
)

_WARMSTART_SRC = Path(warmstart.__file__).read_text(encoding="utf-8")


class _TrainerStub:
    """A minimal trainer duck-type — only `.loaded_from_full_checkpoint` + `.model` are read."""

    def __init__(self, *, loaded_from_full_checkpoint) -> None:
        self.loaded_from_full_checkpoint = loaded_from_full_checkpoint
        self.model = object()


# ── the personal-path default is DEAD (census) ────────────────────────────────────────────
@pytest.mark.parametrize("needle", ["/home/", "/Users/", "expanduser"])
def test_no_personal_path_in_warmstart_source(needle: str) -> None:
    """Census: warmstart.py carries NO host-coupled personal path / expanduser (the deleted
    `_HEADSWAP_AB_DIR = "/home/…"` default). Bites: any personal-path default resurrected."""
    assert needle not in _WARMSTART_SRC, f"warmstart.py must not contain {needle!r} (R1 host-coupling ban)"


def test_no_absolute_head_dir_default() -> None:
    """The old `_HEADSWAP_AB_DIR` absolute default name is gone, and the arm mapping is RELATIVE."""
    assert "_HEADSWAP_AB_DIR" not in _WARMSTART_SRC
    for rel in HEAD_FILE_BY_TYPE.values():
        assert not rel.startswith("/"), f"head file mapping {rel!r} must be relative, not absolute"
    for head_type in ("scalar", "dist65"):
        rel = default_head_for_arm(head_type)
        assert not rel.startswith("/") and "/home/" not in rel


# ── head_dir is REQUIRED ──────────────────────────────────────────────────────────────────
def test_resolve_warmstart_head_file_requires_head_dir() -> None:
    """`resolve_warmstart_head_file(head_dir, value_head_type)` takes head_dir as a required
    positional — there is no code-side default (calling without it is a TypeError)."""
    with pytest.raises(TypeError):
        resolve_warmstart_head_file(value_head_type="scalar")  # type: ignore[call-arg]


def test_resolve_warmstart_head_file_unknown_type_raises() -> None:
    with pytest.raises(ValueError):
        resolve_warmstart_head_file("some/head/dir", "not_a_head_type")


def test_resolve_warmstart_head_file_missing_file_raises_loud(tmp_path: Path) -> None:
    """A misconfigured head_dir (no head file present) fails LOUDLY at launch, never silently
    seeds nothing."""
    with pytest.raises(FileNotFoundError):
        resolve_warmstart_head_file(str(tmp_path), "scalar")


def test_maybe_warmstart_missing_head_dir_raises_loud() -> None:
    """warm_start.enabled with head_dir UNSET → a loud ValueError naming head_dir (never a
    host-coupled default)."""
    trainer = _TrainerStub(loaded_from_full_checkpoint=False)
    with pytest.raises(ValueError, match="head_dir"):
        maybe_warmstart_value_head(trainer, {"warm_start": {"enabled": True}})


def test_maybe_warmstart_disabled_is_noop() -> None:
    """Default-OFF: no warm_start section → a byte-identical no-op (returns False, touches nothing)."""
    trainer = _TrainerStub(loaded_from_full_checkpoint=False)
    assert maybe_warmstart_value_head(trainer, {}) is False
    assert maybe_warmstart_value_head(trainer, {"warm_start": {"enabled": False}}) is False


def test_maybe_warmstart_full_resume_skips() -> None:
    """RESUME GUARD: a full-checkpoint resume already restored the trained head — the hook skips
    (returns False) even when warm_start is enabled (no re-seed corruption)."""
    trainer = _TrainerStub(loaded_from_full_checkpoint=True)
    assert maybe_warmstart_value_head(
        trainer, {"warm_start": {"enabled": True, "head_dir": "some/dir"}, "value_head_type": "scalar"}
    ) is False
