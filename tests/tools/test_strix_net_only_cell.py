"""The NET-ONLY cell: strix with its root VCF solver OFF, threaded from the follower's unit to the driver's load."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from mantis.bots.protocol import BotProtocol
from mantis.bots.strix import NET_ONLY_SUFFIX, StrixBot, load_request
from mantis.config.census import production_configs
from _toolpath import load_module_by_path

_REPO = Path(__file__).resolve().parents[2]
_STEM = "checkpoint_00237000"


def _load(name: str, rel: str) -> Any:
    return load_module_by_path(name, _REPO / rel)


@pytest.fixture(scope="module")
def follower():
    return _load("strix_follower_net_only", "tools/strix_follower.py")


@pytest.fixture(scope="module")
def frontier():
    return _load("strength_frontier_net_only", "tools/strength_frontier.py")


def test_the_load_request_carries_the_solver_switch_and_the_default_is_on() -> None:
    """Nothing on record changes: the pinned variant loads with the solver ON, as every reading was made."""
    on = load_request("/ck.pt", sims=256, variant=_STEM, stem=_STEM)
    off = load_request("/ck.pt", sims=256, variant=_STEM + NET_ONLY_SUFFIX, stem=_STEM)
    assert on["disable_forcing_solver"] is False and off["disable_forcing_solver"] is True
    assert on["sims"] == off["sims"] == 256 and on["m_actions"] == off["m_actions"]


def test_the_net_only_variant_is_a_distinct_instrument_by_name() -> None:
    class _T:
        def ask(self, request: dict[str, Any]) -> dict[str, Any]:
            return {}

        def close(self) -> None:
            pass

    on = StrixBot(transport=_T(), name=f"strix_{_STEM}_s256")
    off = StrixBot(transport=_T(), name=f"strix_{_STEM}_s256_nosolver")
    assert isinstance(off, BotProtocol) and on.name != off.name


def test_the_resolver_admits_the_net_only_variant_and_refuses_a_third() -> None:
    from mantis.bots.strix import RungUnresolvable, variant_solver

    assert variant_solver(_STEM, stem=_STEM) is True
    assert variant_solver(_STEM + NET_ONLY_SUFFIX, stem=_STEM) is False
    with pytest.raises(RungUnresolvable):
        variant_solver("checkpoint_00000001", stem=_STEM)


@pytest.mark.parametrize("config_path", production_configs(_REPO), ids=lambda p: p.name)
def test_a_frontier_cell_with_the_solver_off_composes_the_net_only_rung(
        frontier, tmp_path: Path, config_path: Path) -> None:
    from mantis.config.loader import load_config

    config = load_config(config_path)
    base = frontier.base_round_spec(config, work_dir=tmp_path / "w")
    cell = {"label": "net_only", "candidate": "x", "search_kind": "puct", "sims": 256, "opponent": "strix",
            "strix_sims": 256, "strix_solver": False, "games": 288, "concurrency": 8}
    job = frontier.cell_spec(cell, base, cell_dir=tmp_path / "c", config=config).rung_jobs[0]
    assert job.variant == _STEM + NET_ONLY_SUFFIX and job.opponent_sims == 256
    on = dict(cell, strix_solver=True)
    assert frontier.cell_spec(on, base, cell_dir=tmp_path / "d", config=config).rung_jobs[0].variant == _STEM
    default = {k: v for k, v in cell.items() if k != "strix_solver"}
    assert frontier.cell_spec(default, base, cell_dir=tmp_path / "e", config=config).rung_jobs[0].variant == _STEM


def test_the_follower_unit_composes_the_cell_and_names_its_sidecar(follower) -> None:
    assert follower.UNITS["net_only"] == (256, 256, "strix256_nosolver")
    cell = follower.compose_cell(Path("/x/run8_00042000_deadbeef.ckpt"), unit="net_only", step=42000, games=288,
                                 concurrency=8, label="net_only_run8_42000")
    assert (cell["sims"], cell["strix_sims"], cell["strix_solver"]) == (256, 256, False)
    equal = follower.compose_cell(Path("/x/run8_00042000_deadbeef.ckpt"), unit="equal_work", step=42000,
                                  games=288, concurrency=8, label="equal_work_run8_42000")
    assert "strix_solver" not in equal, "the solver-ON units are byte-identical to before"
    assert follower.sidecar_path(Path("/x/a.ckpt"), "net_only").name == "a.ckpt.strix256_nosolver.json"


def test_the_sidecar_records_the_solver_state_and_the_dashboard_labels_it(follower, external, tmp_path: Path) -> None:
    ckpt = tmp_path / "run8_00042000_deadbeef.ckpt"
    ckpt.write_bytes(b"w")
    record = {"label": "l", "cell": {"step": 42000, "concurrency": 8}, "rc": 0, "wall_sec": 1.0,
              "provenance": {"candidate": {"net_hash": "n"}}, "readout": {"wr": 0.2, "wr_ci_lower": 0.1,
                                                                            "wr_ci_upper": 0.3, "games": 288, "eff_n": 288}}
    body = follower.sidecar_record(ckpt, unit="net_only", trigger="once", record=record, regime_name="CONTENDED",
                                   regime_evidence={}, run_id="run8", started=0.0, finished=1.0, strix_pin={})
    assert body["strix"]["solver"] == "off" and body["strix"]["sims"] == 256
    on = follower.sidecar_record(ckpt, unit="equal_work", trigger="once", record=record, regime_name="IDLE",
                                 regime_evidence={}, run_id="run8", started=0.0, finished=1.0, strix_pin={})
    assert on["strix"]["solver"] == "on"
    point = external.parse_sidecar(Path("x.json"), json.loads(json.dumps(body)))
    assert point is not None and point.unit_label == "run8 · net_only: ours PUCT-256 vs strix 256 sims, solver OFF"
    assert external.parse_sidecar(Path("y.json"), on).unit_label == "run8 · equal_work: ours PUCT-256 vs strix 256 sims"


def test_follow_reads_the_equal_work_unit_only(follower) -> None:
    with pytest.raises(SystemExit):
        follower.main(["--config", "c", "--run-dir", "d", "--run-id", "r", "--work-dir", "w", "--follow",
                       "--unit", "net_only"])
