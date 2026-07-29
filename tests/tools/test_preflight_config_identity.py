"""F-B1 closure — parent-side oracles (WPCLEAN Phase RES; LAW-07 both directions).

The gap (WPMINT Phase B, carried as the preflight-report contract's named gap): parent and
child loaded the config independently and only the parent's identity was published, so a
child that read a DIFFERENT file was invisible in the evidence artifact. The closure gives
the parent three honest verdicts over the child's `run_boot_identity` event — match /
mismatch / unwitnessed — with mismatch a NAMED rc-14 failure. This file drives the verdict
function and the rc contract; the child-side producer arm lives in
`tests/test_run_composition.py` (compose_run emits before pool.start), and the sha function
itself is pinned to ONE authority (`mantis.config.loader.config_identity_sha256`).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = REPO_ROOT / "tools" / "ci_gates" / "preflight_mint.py"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("preflight_mint_identity_test", GATE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_matching_identity_is_a_match(tool):
    sha = "a" * 64
    events = [{"event": "run_boot_identity", "run_id": "r", "config_sha256": sha}]
    assert tool.child_config_identity(events, parent_sha=sha) == (sha, "match")


def test_differing_identity_is_a_mismatch_never_rounded_to_match(tool):
    events = [{"event": "run_boot_identity", "config_sha256": "b" * 64}]
    child_sha, verdict = tool.child_config_identity(events, parent_sha="a" * 64)
    assert (child_sha, verdict) == ("b" * 64, "mismatch")


def test_no_identity_event_is_unwitnessed_not_a_silent_match(tool):
    """The residual the contract discloses: a child dead before its sink exists publishes
    nothing — that must read as `unwitnessed`, never as an assumed match."""
    events = [{"event": "heartbeat_watchdog_fired", "reason": "x"}]
    assert tool.child_config_identity(events, parent_sha="a" * 64) == (None, "unwitnessed")


def test_the_last_identity_event_wins(tool):
    """Segments are run_id-scoped and appended in boot order; the burst under audit is the
    LAST boot."""
    events = [
        {"event": "run_boot_identity", "config_sha256": "0" * 64},
        {"event": "run_boot_identity", "config_sha256": "a" * 64},
    ]
    assert tool.child_config_identity(events, parent_sha="a" * 64) == ("a" * 64, "match")


def test_mismatch_error_class_carries_rc_14(tool):
    """The rc contract: identity mismatch is a NAMED preflight failure in the free 14-19
    band, distinct from every boot/assertion/timeout class."""
    assert issubclass(tool.PreflightConfigIdentityError, tool.PreflightError)
    assert tool.PreflightConfigIdentityError.rc == 14
    taken = {cls.rc for name in dir(tool)
             if isinstance((cls := getattr(tool, name)), type)
             and issubclass(cls, tool.PreflightError) and cls is not tool.PreflightConfigIdentityError}
    assert 14 not in taken, "rc 14 must belong to the identity class alone"


def test_the_one_authority_hashes_identically_for_identical_configs(tool):
    """Both sides call `config_identity_sha256`; two loads of the same file must agree and
    a one-key change must not (the discriminating negative)."""
    from mantis.config.loader import config_identity_sha256, load_config

    path = REPO_ROOT / "configs" / "smoke_preflight_armed.yaml"
    a, b = load_config(path), load_config(path)
    assert config_identity_sha256(a) == config_identity_sha256(b)
    mutated = a.model_copy(update={"seed": int(a.seed) + 1})
    assert config_identity_sha256(mutated) != config_identity_sha256(a)
