"""An xdist worker takes its share of the cores and its own repo probe paths; a serial run takes neither."""
from __future__ import annotations

import pytest
from _xdist_share import apply_thread_share, thread_share, worker_suffix


@pytest.mark.parametrize(
    ("workers", "cpus", "share"),
    [(None, 16, None), (8, 16, 2), (4, 16, 4), (3, 16, 5), (32, 16, 1), (1, 16, 16)],
)
def test_the_share_divides_the_cores_among_the_workers(workers, cpus, share) -> None:
    assert thread_share(workers, cpus) == share


def test_a_worker_exports_its_share_to_every_pool_its_children_read() -> None:
    env: dict[str, str] = {"PYTEST_XDIST_WORKER_COUNT": "8"}
    assert apply_thread_share(env, cpus=16) == 2
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TORCH_INTEROP_THREADS"):
        assert env[var] == "2", var


def test_an_explicit_thread_count_is_never_overridden() -> None:
    env = {"PYTEST_XDIST_WORKER_COUNT": "8", "OMP_NUM_THREADS": "5"}
    apply_thread_share(env, cpus=16)
    assert env["OMP_NUM_THREADS"] == "5"


def test_a_serial_run_exports_nothing() -> None:
    env: dict[str, str] = {}
    assert apply_thread_share(env, cpus=16) is None
    assert env == {}


def test_each_worker_names_its_own_probe_paths() -> None:
    assert worker_suffix({"PYTEST_XDIST_WORKER": "gw3"}) == ".gw3"
    assert worker_suffix({}) == ""
