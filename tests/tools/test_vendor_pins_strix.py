"""The strix pin (RUNG-2): the commit, the checkpoint's sha256, the embedded-config sha256, the disclosed gap."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_PINS = _REPO / "vendor" / "pins.toml"
_BUILD = _REPO / "tools" / "vendor_build_strix.sh"
_DRIVER = _REPO / "tools" / "strix_driver.py"

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: Upstream `main` on 2026-09-13 (`git ls-remote`), the day the checkpoint was supplied.
_STRIX_SHA = "5a771e572553a8bd8e010112b2ce65f16e5afa1b"
#: The operator's checkpoint, hashed at contact.
_CHECKPOINT_SHA256 = "351ed562065bed55528bf4a4bcc7da7bf48f36aec47c60ab108014d6a256447e"


def _pin() -> dict:
    return tomllib.loads(_PINS.read_text(encoding="utf-8"))["pins"]["hexo-strix"]


def test_the_strix_pin_is_the_recorded_commit_at_a_public_url() -> None:
    pin = _pin()
    assert _SHA_RE.match(pin["sha"]) and pin["sha"] == _STRIX_SHA
    assert pin["url"].startswith("https://github.com/SootyOwl/hexo-strix")
    assert "patch" not in pin, "strix is played through its own API; no patch is applied"


def test_the_pin_names_the_checkpoint_its_sha256_and_the_embedded_config_sha256() -> None:
    pin = _pin()
    assert pin["checkpoint"] == "checkpoint_00237000.pt"
    assert pin["checkpoint_sha256"] == _CHECKPOINT_SHA256 and _SHA256_RE.match(pin["checkpoint_sha256"])
    assert pin["checkpoint_train_steps"] == 237000
    assert _SHA256_RE.match(pin["config_sha256"])
    assert "no config.toml was supplied" in pin["config"], "the gap is disclosed in the pin itself"


def test_the_build_script_and_the_driver_are_tracked_and_the_script_verifies_the_sha() -> None:
    text = _BUILD.read_text(encoding="utf-8")
    assert 'pins["hexo-strix"]["sha"]' in text and "uv sync --group cpu" in text
    assert _DRIVER.is_file()
    assert "import mantis" not in _DRIVER.read_text(encoding="utf-8"), "the driver runs in strix's venv"
