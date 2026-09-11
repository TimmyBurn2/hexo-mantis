"""The R348(c) preflight stamp: named refusals, host-state store, tree-bound acceptance."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mantis.config.loader import config_identity_sha256, load_config
from mantis.config.preflight_stamp import (
    START_HALT_READINGS,
    PreflightStampMalformedError,
    PreflightStampMissingError,
    PreflightStampUnmirroredError,
    PreflightStampRefusal,
    PreflightStampTreeMismatchError,
    clear_stamp,
    read_stamp,
    require_preflight_stamp,
    stamp_dir,
    stamp_path,
    write_stamp,
)
from mantis.util.git import head_sha

_REPO = Path(__file__).resolve().parents[2]
_CONFIG = _REPO / "configs" / "run6.yaml"
_HALTS = {"workspace": {"verdict": "MIRRORED", "run_dir": "/x", "bundle": {"step": 1, "files": {}},
                        "shard": {"name": "s", "sha256": ""}},
          "cuda_build": {"verdict": "not_run"}}


@pytest.fixture
def state_home(monkeypatch, tmp_path) -> Path:
    home = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(home))
    return home


def _write(tmp_path: Path, **over: Any):
    config = load_config(_CONFIG)
    kwargs: dict[str, Any] = dict(config=config, config_path=_CONFIG, tree_root=_REPO,
                                  halts=_HALTS, booted_config_sha256="booted", burst_steps=16,
                                  report_path=tmp_path / "report.json")
    kwargs.update(over)
    return config, write_stamp(**kwargs)


def test_the_store_is_host_state_under_xdg_never_the_tree(state_home: Path) -> None:
    assert stamp_dir() == state_home / "mantis" / "preflight"
    assert _REPO not in stamp_dir().parents and stamp_dir() != _REPO


def test_a_passing_stamp_on_this_tree_is_accepted_and_carries_the_halts(
    state_home: Path, tmp_path: Path,
) -> None:
    config, path = _write(tmp_path)
    assert path == stamp_path(config_identity_sha256(config)) and path.is_file()
    stamp = require_preflight_stamp(config, tree_root=_REPO)
    assert stamp["tree_sha"] == head_sha(_REPO) is not None
    assert stamp["halts"] == _HALTS and stamp["verdict"] == "pass"
    assert stamp["run_id"] == config.run_id and stamp["burst_steps"] == 16
    assert stamp["preflight_utc"].endswith("Z")


def test_no_stamp_is_a_named_refusal(state_home: Path) -> None:
    config = load_config(_CONFIG)
    with pytest.raises(PreflightStampMissingError, match=config_identity_sha256(config)):
        require_preflight_stamp(config, tree_root=_REPO)


def test_a_stamp_from_another_tree_is_refused_by_name(state_home: Path, tmp_path: Path) -> None:
    config, path = _write(tmp_path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["tree_sha"] = "0" * 40
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(PreflightStampTreeMismatchError, match="0" * 40):
        require_preflight_stamp(config, tree_root=_REPO)


def test_a_launch_outside_any_checkout_cannot_be_covered(state_home: Path, tmp_path: Path) -> None:
    config, _ = _write(tmp_path)
    assert head_sha(tmp_path) is None, "premise: tmp_path is not inside a git checkout"
    with pytest.raises(PreflightStampTreeMismatchError, match="no readable HEAD"):
        require_preflight_stamp(config, tree_root=tmp_path)


def test_a_stamp_filed_under_another_identity_is_malformed(state_home: Path, tmp_path: Path) -> None:
    config, path = _write(tmp_path)
    other = stamp_path("f" * 64)
    other.write_bytes(path.read_bytes())
    with pytest.raises(PreflightStampMalformedError, match="filed under"):
        read_stamp("f" * 64)
    assert require_preflight_stamp(config, tree_root=_REPO)["config_sha256"] != "f" * 64


@pytest.mark.parametrize("missing", START_HALT_READINGS)
def test_a_stamp_without_a_start_halt_reading_is_refused_at_write(
    state_home: Path, tmp_path: Path, missing: str,
) -> None:
    halts = {k: v for k, v in _HALTS.items() if k != missing}
    with pytest.raises(PreflightStampMalformedError, match=missing):
        _write(tmp_path, halts=halts)
    config = load_config(_CONFIG)
    assert not stamp_path(config_identity_sha256(config)).exists(), "a refused write left a file"


@pytest.mark.parametrize("edit", [
    {"verdict": "fail"}, {"schema_version": 0}, {"halts": {"workspace": {}}},
])
def test_a_stamp_that_stopped_saying_pass_is_malformed(
    state_home: Path, tmp_path: Path, edit: dict,
) -> None:
    config, path = _write(tmp_path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc.update(edit)
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(PreflightStampMalformedError):
        require_preflight_stamp(config, tree_root=_REPO)


def test_unparseable_bytes_are_malformed_not_a_crash(state_home: Path, tmp_path: Path) -> None:
    config, path = _write(tmp_path)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(PreflightStampMalformedError, match="not JSON"):
        require_preflight_stamp(config, tree_root=_REPO)


def test_clear_stamp_removes_the_pass_and_is_idempotent(state_home: Path, tmp_path: Path) -> None:
    config, path = _write(tmp_path)
    clear_stamp(config)
    assert not path.exists()
    clear_stamp(config)
    with pytest.raises(PreflightStampMissingError):
        require_preflight_stamp(config, tree_root=_REPO)


def test_a_reading_without_the_mirrored_verdict_covers_no_host(state_home: Path, tmp_path: Path) -> None:
    """R349(b): any workspace verdict but MIRRORED (the deleted arm's DURABLE included) is refused."""
    unmirrored = {"workspace": {"verdict": "DURABLE", "fstype": "xfs"},
                  "cuda_build": {"verdict": "not_run"}}
    config, _path = _write(tmp_path, halts=unmirrored)
    with pytest.raises(PreflightStampUnmirroredError, match="never proven"):
        require_preflight_stamp(config, tree_root=_REPO)


def test_every_refusal_is_one_named_family() -> None:
    for cls in (PreflightStampMissingError, PreflightStampMalformedError,
                PreflightStampTreeMismatchError, PreflightStampUnmirroredError):
        assert issubclass(cls, PreflightStampRefusal)
