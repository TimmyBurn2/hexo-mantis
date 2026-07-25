"""Suite F-06 — `load_state_dict_safe` must not race with an in-flight forward.

IMPL-written (non-⊕) port of the old race regression. The swap takes `_weights_lock`, so
it blocks until any in-flight forward completes; afterwards every parameter must
BYTE-EQUAL the new weights — a torn write here would silently mix two weight epochs into
one forward and show up only as unexplained strength noise.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import torch

from mantis.encoding import lookup
from mantis.model import CnnArch, build_net
from mantis.selfplay.inference_server import InferenceServer

_SPEC = lookup("v6")
BOARD_CHANNELS = _SPEC.n_planes
BOARD_SIZE = _SPEC.trunk_size


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_model(device: torch.device, seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    net = build_net(
        CnnArch(
            board_size=BOARD_SIZE, in_channels=BOARD_CHANNELS, filters=32, res_blocks=1,
        )
    ).to(device)
    net.eval()
    return net


def _random_state() -> np.ndarray:
    return np.random.randn(BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float16)


def test_load_state_dict_during_forward_is_safe(device: torch.device) -> None:
    model_a = _make_model(device, seed=1)
    model_b = _make_model(device, seed=2)

    params_b = {k: v.clone() for k, v in model_b.state_dict().items()}
    assert not all(
        torch.equal(model_a.state_dict()[k], params_b[k]) for k in params_b
    ), "model_a and model_b must differ — otherwise the test is vacuous"

    server = InferenceServer(
        model_a, device,
        # WPSC Phase 2 SC-A2: InferenceHParams.from_config reads config["inference"] now.
        # WPSC Phase 3 SC-B2: resolve_from_config requires an explicit 'encoding' key
        # (R28) — this fixture's model is v6-grid-derived, so it is spelled out here.
        # WPSC Phase 3 SC-B3: InferenceServer now hard-reads config["train"]["amp_dtype"]
        # unconditionally (R30b, no fallback).
        {"inference": {
            "inference_batch_size": 4, "inference_max_wait_ms": 50.0,
            "trace_inference": True, "compile_inference": False,
            "compile_inference_mode": "default", "compile_inference_dynamic": True,
            "perf_timing": False, "perf_sync_cuda": False,
        }, "encoding": "v6", "train": {"amp_dtype": "fp16"}},
    )
    server.start()

    errors: list[str] = []
    infer_lock = threading.Lock()
    infer_results: list[tuple[np.ndarray, float]] = []
    n_infers = 20

    def _infer() -> None:
        try:
            policy, value = server.infer(_random_state())
            with infer_lock:
                infer_results.append((policy, value))
        except Exception as exc:  # noqa: BLE001 — recorded for the assertion below
            with infer_lock:
                errors.append(f"infer failed: {exc}")

    def _swap() -> None:
        # Small sleep so a forward starts first and the swap races into the lock.
        time.sleep(0.005)
        try:
            server.load_state_dict_safe(params_b)
        except Exception as exc:  # noqa: BLE001 — recorded for the assertion below
            errors.append(f"load_state_dict_safe failed: {exc}")

    threads = [threading.Thread(target=_infer, daemon=True) for _ in range(n_infers)]
    swap_thread = threading.Thread(target=_swap, daemon=True)

    for t in threads:
        t.start()
    swap_thread.start()
    for t in threads:
        t.join(timeout=30.0)
    swap_thread.join(timeout=10.0)

    server.stop()
    server.join(timeout=3.0)

    assert errors == [], "race errors:\n" + "\n".join(errors)
    assert len(infer_results) == n_infers

    final_sd = server.model.state_dict()
    for k, expected in params_b.items():
        actual = final_sd[k]
        assert torch.equal(actual, expected), (
            f"param {k!r} torn after a concurrent load_state_dict_safe"
        )
