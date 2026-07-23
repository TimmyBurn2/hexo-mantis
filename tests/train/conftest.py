"""Shared fixtures for the WP10 ⊕⊕ conformance suites (tests/train/).

This conftest imports ONLY already-present layers (torch + mantis.model / mantis.encoding
/ mantis.config) and NEVER `mantis.train.*` — so it collects cleanly while the two suites
are RED (the suites import `mantis.train.*`, which does not exist until IMPL; that is the
correct oracle-first state). Helper spies (EventSink / clock / call recorders) are plain
duck-typed classes: the injected `EventSink` is a structural Protocol (single `emit` method),
so a bare class with `.emit` satisfies it without importing the not-yet-written Protocol.

Root conftest already installs the autouse `_reseed` fixture (random/numpy/torch) — this
file does NOT re-seed and does NOT touch sys.modules (R5/LAW-17).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from mantis.encoding import lookup
from mantis.model import CnnArch, arch_from_spec_and_config, build_net

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
TRAIN_FIXTURES = FIXTURES / "train"
ANCHOR_KEYS_FILE = FIXTURES / "value_probes" / "anchor_keys" / "v6_live2.txt"

# Encoding used for the grid checkpoint tests: v6_live2_ls (grid, 19, 4 planes) is the O3b
# PASS anchor lineage and a registered encoding.
GRID_ENCODING = "v6_live2_ls"
KILLED_PREFIXES = ("cluster_pool.", "global_encoder.", "gpool_bias_branch.")


# ── spies (duck-typed; satisfy the structural EventSink / callable seams) ─────────────────
class SpyEventSink:
    """Records every emitted event Mapping. Satisfies the structural `EventSink` Protocol
    (single `emit(event: Mapping)` method). The event NAME travels under the `event` key
    (mantis emit convention; cf. `mantis.config.emit.ResolvedConfig.to_event_payload`)."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event") == name]

    def has(self, name: str) -> bool:
        return any(e.get("event") == name for e in self.events)


class FakeClock:
    """Controllable monotonic clock: `clock()` returns the current fake time `t`."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def spy_sink() -> SpyEventSink:
    return SpyEventSink()


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


# ── tiny nets + optim/scaler/scheduler (real torch objects) ──────────────────────────────
def make_tiny_arch() -> CnnArch:
    """The DESIGN §b tiny net: build_net(CnnArch(filters=16, res_blocks=1, ...))."""
    return CnnArch(board_size=19, in_channels=4, filters=16, res_blocks=1)


@pytest.fixture
def tiny_arch() -> CnnArch:
    return make_tiny_arch()


@pytest.fixture
def tiny_net(tiny_arch: CnnArch) -> torch.nn.Module:
    return build_net(tiny_arch)


def make_optim_scaler_sched(
    net: torch.nn.Module, *, lr: float = 1e-3, t_max: int = 1000, eta_min: float = 1e-5
):
    """Two-param-group AdamW (weight-decay split → the golden's `param_groups==2`) + a CPU
    GradScaler with real state + a CosineAnnealingLR."""
    decay = [p for _, p in net.named_parameters() if p.ndim >= 2]
    no_decay = [p for _, p in net.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 1e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )
    scaler = torch.amp.GradScaler("cpu", enabled=True)  # non-empty state_dict on CPU
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=eta_min)
    return opt, scaler, sched


@pytest.fixture
def optim_scaler_sched(tiny_net: torch.nn.Module):
    return make_optim_scaler_sched(tiny_net)


# ── full v6_live2_ls net (registry arch) — the strict-load target for legacy / O3b tests ──
# The bare O3b anchor and the legacy read path resolve arch from the encoding → the FULL
# registry arch (filters=128, res_blocks=12), NOT a tiny net. Built once per session.
@pytest.fixture(scope="session")
def full_ls_net() -> torch.nn.Module:
    return build_net(arch_from_spec_and_config(lookup(GRID_ENCODING), {}))


@pytest.fixture
def full_ls_state(full_ls_net: torch.nn.Module) -> dict[str, torch.Tensor]:
    """A fresh shallow copy of the full v6_live2_ls state dict (147 keys, O3b-clean)."""
    return dict(full_ls_net.state_dict())


# ── schema-valid / invalid config snapshots (validated against config-schema v1 on write) ─
def make_run_config(encoding: str = GRID_ENCODING, representation: str = "grid",
                    run_id: str = "run5") -> dict[str, Any]:
    """A complete, schema-v1-valid RunConfig dict (the envelope `config` snapshot)."""
    return {
        "schema_version": 1,
        "run_id": run_id,
        "seed": 20260718,
        "identity": {"encoding": encoding, "representation": representation},
        "eval": {"random_model_sims": 96, "sealbot_model_sims": 128},
        "selfplay": {"legal_move_radius_schedule": None},
    }


@pytest.fixture
def valid_config() -> dict[str, Any]:
    return make_run_config()


@pytest.fixture
def invalid_config() -> dict[str, Any]:
    """A config that fails schema v1 (extra=forbid): a complete config + one unknown key."""
    cfg = make_run_config()
    cfg["__unknown_knob__"] = True
    return cfg


# ── metadata_kwargs (the stamp inputs to save_checkpoint; encoding_name REQUIRED) ─────────
def make_metadata_kwargs(arch: CnnArch, *, encoding_name: str = GRID_ENCODING,
                         run_id: str = "runa", corpus_sha256: str | None = None
                         ) -> dict[str, Any]:
    """The metadata stamp inputs. `created_utc`/`commit_sha` are stamped ONCE by
    save_checkpoint (NOT supplied here — supplying them is the restamp error, T-CK-10)."""
    mk: dict[str, Any] = {"encoding_name": encoding_name, "run_id": run_id, "arch": arch}
    if corpus_sha256 is not None:
        mk["corpus_sha256"] = corpus_sha256
    return mk


@pytest.fixture
def metadata_kwargs(tiny_arch: CnnArch) -> dict[str, Any]:
    return make_metadata_kwargs(tiny_arch)


# ── factory fixtures (callables, so a test can vary encoding/run_id without importing
#    conftest by name — R5/LAW-17 keeps the collection style import-hack-free) ─────────────
@pytest.fixture
def mk_config():
    return make_run_config


@pytest.fixture
def mk_meta():
    return make_metadata_kwargs


@pytest.fixture
def mk_optim():
    return make_optim_scaler_sched


# ── committed goldens (manifest-tracked) ─────────────────────────────────────────────────
@pytest.fixture(scope="session")
def resume_goldens() -> dict[str, Any]:
    return json.loads((TRAIN_FIXTURES / "resume_goldens.json").read_text())


@pytest.fixture(scope="session")
def legacy_shapes() -> dict[str, Any]:
    return json.loads((TRAIN_FIXTURES / "legacy_payload_shapes.json").read_text())


@pytest.fixture(scope="session")
def anchor_key_set() -> set[str]:
    return set(ANCHOR_KEYS_FILE.read_text().split())
