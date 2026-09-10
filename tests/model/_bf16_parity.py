"""Shared instrument for the WP12-R F1 bf16 parity oracles.

NOT COLLECTED (leading `_`): it carries the one fixture, the one arm runner and the pinned
null-distribution artifact, so the drift rows and the null-calibration rows read the SAME two
arms and the SAME grounds. R8: 300-line soft cap not exceeded.

THE PINNED ARTIFACT is the box measurement of the HEAD-vs-HEAD distribution of every
OF1-3/OF1-4 statistic on CUDA at commit 982da03, torch 2.11.0+cu128, RTX 5080 — 3675 null
pairs, 3000 F1-vs-HEAD pairs, 1140 CPU null pairs, 1200 CPU F1-vs-HEAD pairs. IT IS
DEVICE-SPECIFIC: the median form's exact zero there is a property of the 5080's kernels, and an
RTX 4060 (sm_89, cu130) reads 0/15 pairs zero, worst 1.395037e-02 — above F1's own maximum
effect (1.365076e-02). So the CUDA parity leg asserts EXACT equality under determinism, and
PER-DEVICE ENVELOPE CALIBRATION IS REJECTED as a treadmill.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from mantis.model.arch import GnnArch
from mantis.model.dist65 import binned_value_loss
from mantis.model.gnn import GnnNet
from mantis.train.losses import ragged_policy_ce

NULLDIST_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "bf16_nulldist"
    / "measurement_raw_R181_NULLDIST.json"
)
NULLDIST_SHA256 = "b149659d1d423a05caf55a86371183e51dbe3d7b8a5d4249494b809d19dc72fb"


class NullDistArtifactError(Exception):
    """Raised when the pinned artifact is absent or its sha256 has drifted. FAILS, never skips:
    grounds that cannot be read are not grounds."""


def load_nulldist(path: Path = NULLDIST_PATH) -> dict:
    """Read and sha-verify the pinned artifact. Raises `NullDistArtifactError`, never skips.

    `path` exists so the producer test can drive THIS call path on a `tmp_path` copy;
    `NULLDIST_SHA256` is NOT a parameter, or the self-test could pass against a sha the caller
    chose. ORDER IS LOAD-BEARING: the sha gate runs BEFORE `json.loads`.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise NullDistArtifactError(f"pinned R181 artifact unreadable: {path}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if digest != NULLDIST_SHA256:
        raise NullDistArtifactError(
            f"pinned R181 artifact sha256 drift: {digest} != {NULLDIST_SHA256}"
        )
    return json.loads(raw.decode())


@contextlib.contextmanager
def deterministic_algorithms():
    """TEST SCOPE ONLY. Enable `torch.use_deterministic_algorithms(True)` for the block, then
    restore the ambient setting exactly — it is process-global, so leaking it would silently
    change the numerics of every sibling test. PRODUCTION KEEPS ITS KERNELS: nothing in
    `src/mantis/` calls this. `CUBLAS_WORKSPACE_CONFIG` is set if absent and restored."""
    was_enabled = torch.are_deterministic_algorithms_enabled()
    had_cublas = "CUBLAS_WORKSPACE_CONFIG" in os.environ
    old_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if not had_cublas:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.use_deterministic_algorithms(was_enabled)
        if had_cublas:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = old_cublas
        else:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def cuda_pairs(doc: dict, side: str, stat: str) -> dict[str, list[float]]:
    """Per-fixture per-pair values of `stat`. `side` is 'null' (identical code) or 'alt'
    (F1-vs-HEAD). Reads the columnar blocks — the raw values, not the report's summaries."""
    key = "null_pairs_treat_columnar" if side == "null" else "alt_pairs_treat_columnar"
    return {fx: c[key][stat] for fx, c in doc["cuda"]["fixtures"].items()}


_SEED = 20260803
_N_GRAPHS = 16
_N_MIN, _N_MAX = 26, 512
_MEAN_IN_DEGREE = 8


@dataclass(frozen=True)
class Batch:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    legal_index: torch.Tensor
    stone_mask: torch.Tensor
    node_offsets: torch.Tensor
    legal_offsets: torch.Tensor
    policy_target: torch.Tensor
    outcomes: torch.Tensor
    value_valid: torch.Tensor
    is_full_search: torch.Tensor
    mean_in_degree: float


def build_arch() -> GnnArch:
    return GnnArch(in_dim=11, edge_dim=5, hidden=128, num_layers=4,
                   policy_hidden=128, value_hidden=32)


def build_net() -> GnnNet:
    """The net PREREG_DFIX §1 fixes, at its pinned seed, in `eval()`. Built through
    `mantis.model.build_net` rather than `GnnNet(...)`, because a direct constructor left the
    parity net without an `.arch` handle. Seed and construction order are unchanged."""
    from mantis.model import build_net as _build_net

    torch.set_num_threads(1)
    torch.manual_seed(_SEED)
    net = _build_net(build_arch())
    assert isinstance(net, GnnNet)
    net.eval()
    return net


def build_batch(arch: GnnArch) -> Batch:
    """A block-diagonal batch of `_N_GRAPHS` graphs, per-graph N in [26, 512] and edges drawn to
    a mean in-degree of ~`_MEAN_IN_DEGREE`, with offsets following the production collate."""
    gen = torch.Generator().manual_seed(_SEED + 1)
    sizes = torch.randint(_N_MIN, _N_MAX + 1, (_N_GRAPHS,), generator=gen).tolist()
    xs, edges, attrs, legal, stone = [], [], [], [], []
    node_offsets = [0]
    legal_offsets = [0]
    policy_targets = []
    base = 0
    for n in sizes:
        xs.append(torch.randn(n, arch.in_dim, generator=gen))
        n_edges = n * _MEAN_IN_DEGREE
        src = torch.randint(0, n, (n_edges,), generator=gen)
        dst = torch.randint(0, n, (n_edges,), generator=gen)
        edges.append(torch.stack((src, dst)) + base)
        attrs.append(torch.randn(n_edges, arch.edge_dim, generator=gen))
        # Roughly half the nodes are legal-move nodes; the exact split is not pinned.
        is_legal = torch.zeros(n, dtype=torch.bool)
        is_legal[: n // 2] = True
        legal.append(is_legal)
        stone.append(~is_legal)
        n_legal = int(is_legal.sum())
        tgt = torch.rand(n_legal, generator=gen)
        policy_targets.append(tgt / tgt.sum())
        base += n
        node_offsets.append(base)
        legal_offsets.append(legal_offsets[-1] + n_legal)
    edge_index = torch.cat(edges, dim=1)
    n_total = base
    return Batch(
        x=torch.cat(xs),
        edge_index=edge_index,
        edge_attr=torch.cat(attrs),
        # The wire's `legal_node_gather` shape: the ROWS of the legal nodes, strictly ascending,
        # derived from the masks this builder already makes, so the SET is unchanged.
        legal_index=torch.nonzero(torch.cat(legal), as_tuple=False).reshape(-1),
        stone_mask=torch.cat(stone),
        node_offsets=torch.tensor(node_offsets, dtype=torch.long),
        legal_offsets=torch.tensor(legal_offsets, dtype=torch.long),
        policy_target=torch.cat(policy_targets),
        outcomes=torch.empty(_N_GRAPHS).uniform_(-1.0, 1.0, generator=gen),
        # Mixed masks on purpose: an all-ones mask would exercise neither masked reduction.
        value_valid=(torch.arange(_N_GRAPHS) % 4 != 0).to(torch.uint8),
        is_full_search=(torch.arange(_N_GRAPHS) % 3 != 0).to(torch.uint8),
        mean_in_degree=edge_index.shape[1] / n_total,
    )


@dataclass(frozen=True)
class Arm:
    policy_logits: torch.Tensor
    bin_logits: torch.Tensor
    policy_loss: float
    value_loss: float
    grads: torch.Tensor


def run_arm(net: GnnNet, batch: Batch, *, autocast_enabled: bool,
            dtype: torch.dtype = torch.bfloat16, device: str = "cpu") -> Arm:
    """One forward + loss + backward mirroring `train_step_from_graph_batch`'s numeric core
    without an optimizer; `torch.autograd.grad` keeps the arms from contaminating each other.
    `dtype` is a parameter ONLY so the mutation condition can inject a real numerics change."""
    params = [p for p in net.parameters() if p.requires_grad]
    with torch.autocast(device_type=device, dtype=dtype, enabled=autocast_enabled):
        policy_logits, _value, bin_logits = net.forward_batch(
            batch.x, batch.edge_index, batch.edge_attr, batch.legal_index,
            batch.stone_mask, node_offsets=batch.node_offsets,
        )
        policy_loss = ragged_policy_ce(
            policy_logits, batch.policy_target, batch.legal_offsets,
            full_search_mask=batch.is_full_search,
        )
        value_loss = binned_value_loss(bin_logits, batch.outcomes, value_mask=batch.value_valid)
        loss = policy_loss + value_loss
    grads = torch.autograd.grad(loss, params, allow_unused=False)
    return Arm(
        policy_logits=policy_logits.detach().to(torch.float32),
        bin_logits=bin_logits.detach().to(torch.float32),
        policy_loss=float(policy_loss.detach()),
        value_loss=float(value_loss.detach()),
        grads=torch.cat([g.reshape(-1).to(torch.float32) for g in grads]),
    )


def rel(treat: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """PREREG_DFIX §1's logit statistic: |delta| / (|a| + 1e-3), the floor keeping a near-zero
    logit from manufacturing an unbounded ratio — which is what destroyed the MAX reduction."""
    return (treat - ref).abs() / (ref.abs() + 1e-3)


def median_form(treat: torch.Tensor, ref: torch.Tensor) -> float:
    """THE RE-POINTED STATISTIC: the MEDIAN reduction of `rel`.

    ITS CUDA NULL IS DEVICE-SPECIFIC — do not quote the next line without the device. On RTX
    5080 / sm_120 / torch 2.11.0+cu128 ONLY: exactly 0.000000e+00 on 3675/3675 identical-code
    pairs; an RTX 4060 / sm_89 / cu130 reads 0/15 zero, worst 1.395037e-02. CPU null is
    bit-identity, 1140/1140; under F1 on the 5080, 7.4900e-3 … 1.3651e-2 over 3000 pairs.
    """
    return float(rel(treat, ref).median())
