"""The RULER-R6 cell (R365 E1): strix at its trained placement_radius 6, threaded from the follower's unit to the driver's load."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from mantis.bots.strix import NET_ONLY_SUFFIX, RADIUS_SUFFIX, RungUnresolvable, load_request, variant_radius, variant_solver

_REPO = Path(__file__).resolve().parents[2]
_STEM = "checkpoint_00237000"


def _load(name: str, rel: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def follower():
    return _load("strix_follower_ruler_r6", "tools/strix_follower.py")


@pytest.fixture(scope="module")
def frontier():
    return _load("strength_frontier_ruler_r6", "tools/strength_frontier.py")


@pytest.fixture(scope="module")
def external(dashboard):
    import importlib

    return importlib.import_module("dashboard.external")


def test_the_radius_variant_carries_placement_radius_and_every_other_line_is_unchanged() -> None:
    """Only `<stem>:r<N>` puts the key on the wire; the rung on record and the net-only line stay byte-identical."""
    r6 = load_request("/ck.pt", sims=256, variant=_STEM + RADIUS_SUFFIX + "6", stem=_STEM)
    on = load_request("/ck.pt", sims=256, variant=_STEM, stem=_STEM)
    off = load_request("/ck.pt", sims=256, variant=_STEM + NET_ONLY_SUFFIX, stem=_STEM)
    assert r6["placement_radius"] == 6 and r6["disable_forcing_solver"] is False
    assert "placement_radius" not in on and "placement_radius" not in off
    assert {k: v for k, v in r6.items() if k != "placement_radius"} == on


def test_the_resolver_reads_the_radius_and_refuses_a_malformed_one() -> None:
    assert variant_radius(_STEM + RADIUS_SUFFIX + "6", stem=_STEM) == 6
    assert variant_radius(_STEM, stem=_STEM) is None and variant_radius(_STEM + NET_ONLY_SUFFIX, stem=_STEM) is None
    assert variant_solver(_STEM + RADIUS_SUFFIX + "6", stem=_STEM) is True
    for bad in (_STEM + RADIUS_SUFFIX, _STEM + RADIUS_SUFFIX + "0", _STEM + RADIUS_SUFFIX + "x"):
        with pytest.raises(RungUnresolvable):
            variant_radius(bad, stem=_STEM)


def test_a_frontier_cell_with_strix_radius_composes_the_r6_rung_and_refuses_it_beside_solver_off(
        frontier, tmp_path: Path) -> None:
    from mantis.config.loader import load_config

    config = load_config(str(_REPO / "configs" / "run8.yaml"))
    base = frontier.base_round_spec(config, work_dir=tmp_path / "w")
    cell = {"label": "ruler_r6", "candidate": "x", "search_kind": "puct", "sims": 256, "opponent": "strix",
            "strix_sims": 256, "strix_radius": 6, "games": 288, "concurrency": 8}
    job = frontier.cell_spec(cell, base, cell_dir=tmp_path / "c", config=config).rung_jobs[0]
    assert job.variant == _STEM + RADIUS_SUFFIX + "6" and job.opponent_sims == 256
    default = {k: v for k, v in cell.items() if k != "strix_radius"}
    assert frontier.cell_spec(default, base, cell_dir=tmp_path / "d", config=config).rung_jobs[0].variant == _STEM
    with pytest.raises(frontier.FrontierCellError):
        frontier.cell_spec(dict(cell, strix_solver=False), base, cell_dir=tmp_path / "e", config=config)
    assert "strix_radius=6" in frontier.format_row({"label": "l", "cell": cell, "rc": 1, "wall_sec": 1.0})


def test_the_follower_unit_composes_the_cell_and_names_its_sidecar(follower) -> None:
    assert follower.UNITS["ruler_r6"] == (256, 256, "strix256_r6") and follower.RADIUS_UNITS == {"ruler_r6": 6}
    cell = follower.compose_cell(Path("/x/run8_00045000_deadbeef.ckpt"), unit="ruler_r6", step=45000, games=288,
                                 concurrency=8, label="ruler_r6_run8_45000")
    assert (cell["sims"], cell["strix_sims"], cell["strix_radius"]) == (256, 256, 6) and "strix_solver" not in cell
    equal = follower.compose_cell(Path("/x/run8_00045000_deadbeef.ckpt"), unit="equal_work", step=45000,
                                  games=288, concurrency=8, label="equal_work_run8_45000")
    assert "strix_radius" not in equal, "the default-radius units are byte-identical to before"
    assert follower.sidecar_path(Path("/x/a.ckpt"), "ruler_r6").name == "a.ckpt.strix256_r6.json"


def test_the_sidecar_records_the_radius_and_the_dashboard_labels_it(follower, external, tmp_path: Path) -> None:
    ckpt = tmp_path / "run8_00045000_deadbeef.ckpt"
    ckpt.write_bytes(b"w")
    record = {"label": "l", "cell": {"step": 45000, "concurrency": 8}, "rc": 0, "wall_sec": 1.0,
              "provenance": {"candidate": {"net_hash": "n"}}, "readout": {"wr": 0.2, "wr_ci_lower": 0.1,
                                                                            "wr_ci_upper": 0.3, "games": 288, "eff_n": 288}}
    body = follower.sidecar_record(ckpt, unit="ruler_r6", trigger="once", record=record, regime_name="IDLE",
                                   regime_evidence={}, run_id="run8", started=0.0, finished=1.0, strix_pin={})
    assert body["strix"] == {"sims": 256, "solver": "on", "radius": 6}
    on = follower.sidecar_record(ckpt, unit="equal_work", trigger="once", record=record, regime_name="IDLE",
                                 regime_evidence={}, run_id="run8", started=0.0, finished=1.0, strix_pin={})
    assert "radius" not in on["strix"]
    point = external.parse_sidecar(Path("x.json"), json.loads(json.dumps(body)))
    assert point is not None and point.radius == 6
    assert point.unit_label == "run8 · ruler_r6: ours PUCT-256 vs strix 256 sims, strix @ r6"
    assert external.parse_sidecar(Path("y.json"), on).radius is None
