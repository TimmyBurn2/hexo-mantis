"""The Six pin: the repo at its commit, the v1.2.0 Linux engine asset and two networks, each by URL and sha256."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path, PurePosixPath

_REPO = Path(__file__).resolve().parents[2]
_PINS = _REPO / "vendor" / "pins.toml"
_MAKEFILE = _REPO / "Makefile"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RELEASE_URL = "https://github.com/CixMango/Six/releases/download/v1.2.0/Six-1.2.0-linux-x64.tar.gz"

#: SIX-SCOUT's provenance record: every digest matched GitHub's published one at download.
_EXPECTED = {
    "release": "a5803df2edea482504804804be676a35762435b55cce555697f91347be814f7e",
    "engine": "7d06548be5e55c0f07b56736cbf18190255e99a74c4799ecfa98b03a805cf26a",
    "gen0030": "4afcbb113f18ecfd602b28e8da080c71e91f576dea939b68590f255e60cb5a92",
    "gen0455": "a934a8b171cd9a715fcd54ffc3e192c24901f0a7d40caa4216ea9fbc0b074687",
}


def _pin() -> dict:
    return tomllib.loads(_PINS.read_text(encoding="utf-8"))["pins"]["six"]


def test_the_six_pin_is_the_scouted_commit_at_its_public_url() -> None:
    pin = _pin()
    assert pin["url"] == "https://github.com/CixMango/Six.git"
    assert pin["sha"] == "f2b5ec2d4d7ec42e8b655f739e65821808e03698"
    assert "patch" not in pin, "Six is played through its release binary; nothing is patched"


def test_every_asset_is_named_by_url_and_sha256() -> None:
    assets = _pin()["assets"]
    assert set(assets) == set(_EXPECTED)
    for name, sha in _EXPECTED.items():
        assert assets[name]["sha256"] == sha and _SHA256_RE.match(sha), name
        assert assets[name]["url"].startswith("https://github.com/CixMango/Six/releases/download/"), name


def test_the_engine_and_the_release_network_are_members_of_the_v120_linux_asset() -> None:
    assets = _pin()["assets"]
    assert assets["release"]["url"] == _RELEASE_URL and "unpack" in assets["release"]
    for member in ("engine", "gen0455"):
        assert assets[member]["from"] == "release" and assets[member]["url"] == _RELEASE_URL
        assert PurePosixPath(assets[member]["path"]).is_relative_to(assets["release"]["unpack"])
    assert assets["gen0030"]["url"].endswith("/networks/gen-0030.onnx") and "from" not in assets["gen0030"]


def test_every_asset_path_stays_inside_the_pins_own_asset_directory() -> None:
    for name, spec in _pin()["assets"].items():
        parts = PurePosixPath(spec["path"]).parts
        assert parts[0] == "six-assets" and ".." not in parts, name


def test_make_vendor_six_fetches_the_six_pin_with_its_assets() -> None:
    text = _MAKEFILE.read_text(encoding="utf-8")
    assert re.search(r"^vendor\.six:\n\tbash tools/vendor_fetch\.sh six$", text, re.MULTILINE)
