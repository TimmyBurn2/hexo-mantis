"""S on the GPU: strix's own CUDA venv and `device` on the driver's load, threaded from a cell's `strix_device`."""
from __future__ import annotations

from pathlib import Path

import pytest
from _toolpath import load_module_by_path

from mantis.bots.strix import (
    DEVICE_SUFFIX,
    NET_ONLY_SUFFIX,
    RADIUS_SUFFIX,
    load_request,
    variant_device,
    variant_radius,
    variant_solver,
    venv_python,
)
from mantis.config.census import production_configs
from mantis.config.loader import load_config

_REPO = Path(__file__).resolve().parents[2]
_STEM = "checkpoint_00237000"


@pytest.fixture(scope="module")
def follower():
    return load_module_by_path("strix_follower_gpu", _REPO / "tools/strix_follower.py")


@pytest.fixture(scope="module")
def frontier():
    return load_module_by_path("strength_frontier_gpu", _REPO / "tools/strength_frontier.py")


def test_the_gpu_variant_puts_device_on_the_wire_and_every_other_line_is_unchanged() -> None:
    gpu = load_request("/ck.pt", sims=256, variant=_STEM + DEVICE_SUFFIX, stem=_STEM)
    cpu = load_request("/ck.pt", sims=256, variant=_STEM, stem=_STEM)
    assert gpu["device"] == "cuda" and "device" not in cpu
    assert {k: v for k, v in gpu.items() if k != "device"} == cpu
    r6 = load_request("/ck.pt", sims=256, variant=_STEM + RADIUS_SUFFIX + "6" + DEVICE_SUFFIX, stem=_STEM)
    assert r6["placement_radius"] == 6 and r6["device"] == "cuda"


def test_the_device_reads_off_the_variant_beside_the_solver_and_the_radius() -> None:
    assert variant_device(_STEM + DEVICE_SUFFIX) == "cuda" and variant_device(_STEM) == "cpu"
    assert variant_solver(_STEM + DEVICE_SUFFIX, stem=_STEM) is True
    assert variant_solver(_STEM + NET_ONLY_SUFFIX + DEVICE_SUFFIX, stem=_STEM) is False
    assert variant_radius(_STEM + RADIUS_SUFFIX + "6" + DEVICE_SUFFIX, stem=_STEM) == 6


def test_each_device_runs_its_own_venv(tmp_path: Path) -> None:
    assert venv_python(tmp_path, "cpu") == tmp_path / ".venv" / "bin" / "python"
    assert venv_python(tmp_path, "cuda") == tmp_path / ".venv-cuda" / "bin" / "python"


@pytest.mark.parametrize("config_path", production_configs(_REPO)[:1], ids=lambda p: p.name)
def test_a_frontier_cell_names_strix_on_the_gpu_and_refuses_another_device(frontier, tmp_path: Path,
                                                                           config_path: Path) -> None:
    config = load_config(config_path)
    base = frontier.base_round_spec(config, work_dir=tmp_path / "w")
    cell = {"label": "s_gpu", "candidate": "x", "search_kind": "puct", "sims": 256, "opponent": "strix",
            "strix_sims": 256, "strix_device": "cuda", "games": 576, "concurrency": 8}
    job = frontier.cell_spec(cell, base, cell_dir=tmp_path / "c", config=config).rung_jobs[0]
    assert job.variant == _STEM + DEVICE_SUFFIX
    on_record = {k: v for k, v in cell.items() if k != "strix_device"}
    assert frontier.cell_spec(on_record, base, cell_dir=tmp_path / "d", config=config).rung_jobs[0].variant == _STEM
    with pytest.raises(frontier.FrontierCellError, match="strix_device"):
        frontier.cell_spec({**cell, "strix_device": "rocm"}, base, cell_dir=tmp_path / "e", config=config)


def test_the_s_units_of_record_play_strix_on_the_gpu_and_the_research_variants_stay_on_the_cpu(follower) -> None:
    for unit in ("equal_work", "equal_work_arena"):
        cell = follower.compose_cell(Path("/x/a.ckpt"), unit=unit, step=1, games=576, concurrency=8, label="s")
        assert cell["strix_device"] == "cuda", unit
    for unit in ("net_only", "ruler_r6", "as_shipped"):
        cell = follower.compose_cell(Path("/x/a.ckpt"), unit=unit, step=1, games=288, concurrency=8, label=unit)
        assert "strix_device" not in cell, unit
