"""Shared instrument for the WP12-R F1 bf16 parity oracles (OF1-3 / OF1-4 + the R181 re-point).

NOT COLLECTED (leading `_`, the `tests/model/_value_health.py` precedent). It carries the
one fixture, the one arm runner, and the pinned null-distribution artifact, so that the
drift rows and the null-calibration rows read the SAME two arms and the SAME grounds. Split
across two test files they would drift.

R8: 300-line soft cap not exceeded.

WHAT THE PINNED ARTIFACT IS (R69, R181's "cited grounds"):
`tests/fixtures/bf16_nulldist/measurement_raw_R181_NULLDIST.json` is the box measurement of
the HEAD-vs-HEAD (identical-code) distribution of every OF1-3/OF1-4 statistic on CUDA, at
commit 982da03, torch 2.11.0+cu128, RTX 5080 — 3675 null pairs, 3000 F1-vs-HEAD pairs,
1140 CPU null pairs, 1200 CPU F1-vs-HEAD pairs. Companion report:
`wp/WP12R/MEASUREMENT_NULLDIST.md`.

**THE ARTIFACT IS DEVICE-SPECIFIC (R191). Its measurement STANDS; its GENERALITY DOES NOT.**
Its headline property — the median form reading exactly `0.000000e+00` on 3675/3675
identical-code CUDA pairs — is a property of the **RTX 5080 (sm_120, cu128)** kernels, not of
the statistic. Measured on an **RTX 4060 Laptop (sm_89, torch 2.11.0+cu130)**: the same
statistic on the same fixture reads **0/15 pairs zero, worst 1.395037e-02** — 14.0x the
envelope this file used to assert, and above F1's own maximum measured effect (1.365076e-02).

The mechanism is element-level bit-identity, and the two devices sit either side of the
median: a median of exactly zero ENTAILS that at least half the elements are bit-identical
(that is what a median IS), so the 5080's 3675/3675 zeros entail **>= 50%** there; the 4060
measures **12.0% - 19.1%** across mean in-degrees 8 / 15 / 27. Nothing about the statistic
changed — the kernel majority did.

**Consequence, and it is why no envelope appears in the CUDA null leg any more:** the CUDA
parity leg runs under `deterministic_algorithms()` and asserts EXACT equality. Measured under
determinism on the 4060: **15/15 `torch.equal` on all three tensors, element bit-identity
1.0000.** An exact assertion beats any calibrated bound and is device-independent.
**PER-DEVICE ENVELOPE CALIBRATION IS REJECTED (R191): it is a treadmill, and there is
deliberately no hook for it anywhere in this file.**
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

# ── the pinned null-distribution artifact ────────────────────────────────────────────
NULLDIST_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "bf16_nulldist"
    / "measurement_raw_R181_NULLDIST.json"
)
NULLDIST_SHA256 = "b149659d1d423a05caf55a86371183e51dbe3d7b8a5d4249494b809d19dc72fb"


class NullDistArtifactError(Exception):
    """Raised when the pinned R181 artifact is absent or its sha256 has drifted.

    FAILS, never skips (the `tests/test_fixtures_manifest.py` posture). Grounds that cannot
    be read are not grounds.
    """


def load_nulldist(path: Path = NULLDIST_PATH) -> dict:
    """Read + sha-verify the pinned artifact. Raises `NullDistArtifactError`, never skips.

    `path` defaults to the pinned artifact and every production caller uses the default; it
    exists so the LAW-07 producer test can drive THIS call path on a `tmp_path` copy rather
    than monkeypatching a module constant — the `tests/test_fixtures_manifest.py`
    `check_manifest(manifest_path, fixtures_root)` precedent. `NULLDIST_SHA256` is NOT a
    parameter: the pin must not be substitutable, or the self-test could pass against a
    sha the caller chose.

    ORDER IS LOAD-BEARING: the sha gate runs BEFORE `json.loads`, so an artifact that is
    still perfectly valid JSON and semantically identical is refused on its bytes alone.
    `test_loader_raises_on_sha_drift` mutates exactly that case.
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
    """**TEST SCOPE ONLY.** Enable `torch.use_deterministic_algorithms(True)` for the block,
    then restore the ambient setting exactly as it was.

    **PRODUCTION KEEPS ITS KERNELS (R191).** Nothing in `src/mantis/` calls this; run5 trains
    with the fast nondeterministic `index_add_`, which is the whole reason F1 exists. This
    context is an INSTRUMENT for the parity legs, and every leg that uses it says so in its
    own NAME so no reader can conclude the production run is deterministic.

    Restoring matters: `use_deterministic_algorithms` is process-global, so leaking it would
    silently change the numerics of every sibling test that ran afterwards — a test-ordering
    bug that would look like a flaky oracle. `test_test_scope_determinism_does_not_leak`
    is the producer for the restore.

    `CUBLAS_WORKSPACE_CONFIG` is set if absent because cuBLAS raises under determinism
    without it; it is restored (including back to ABSENT) on exit.
    """
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


# ── the fixture, FIXED in PREREG_DFIX §1 before any measurement ──────────────────────
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
    """The net PREREG_DFIX §1 fixes, at its pinned seed. `eval()`: no dropout/BN state.

    AUDIT-1 F-31: built through `mantis.model.build_net`, not `GnnNet(...)` directly. This was
    the ONLY direct net constructor outside `mantis.model` in the tree, so the parity net
    carried no `.arch` handle — it was not the object production builds, while being the object
    a LAW-06 parity claim rests on. Seed and construction order are unchanged, so the pinned
    net is byte-identical; what changes is that it now comes off the one builder.
    """
    from mantis.model import build_net as _build_net

    torch.set_num_threads(1)
    torch.manual_seed(_SEED)
    net = _build_net(build_arch())
    assert isinstance(net, GnnNet)
    net.eval()
    return net


def build_batch(arch: GnnArch) -> Batch:
    """A block-diagonal batch of `_N_GRAPHS` graphs, per-graph N drawn in [26, 512], edges
    drawn so the mean in-degree is ~`_MEAN_IN_DEGREE`. Node/legal/edge offsets follow the
    production collate contract (`graph_collate.py`): per-graph node offsets already
    applied to `edge_index`, `legal_offsets` over the legal-node segments."""
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
        # Roughly half the nodes are legal-move nodes, the rest stones — the shape the
        # axis-graph carries mid-game; the exact split is not a pinned quantity.
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
        # The wire's `legal_node_gather` shape (R284 P-MASK): the ROWS of the legal nodes,
        # strictly ascending, which is what `forward_batch` gathers with. Derived from the
        # per-graph masks this builder already makes, so the SET is unchanged and the LAW-06
        # arms compare the same graphs they compared before.
        legal_index=torch.nonzero(torch.cat(legal), as_tuple=False).reshape(-1),
        stone_mask=torch.cat(stone),
        node_offsets=torch.tensor(node_offsets, dtype=torch.long),
        legal_offsets=torch.tensor(legal_offsets, dtype=torch.long),
        policy_target=torch.cat(policy_targets),
        outcomes=torch.empty(_N_GRAPHS).uniform_(-1.0, 1.0, generator=gen),
        # Mixed masks on purpose: an all-ones mask would exercise neither loss's masked
        # reduction, and both are live on the production route (`core.py:527-529`).
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
    """One forward + loss + backward, mirroring `train_step_from_graph_batch`'s numeric
    core (`core.py:521-533`) without an optimizer. `torch.autograd.grad` is used so the
    arms cannot contaminate each other through `.grad`.

    `dtype` is a parameter ONLY so the R181 mutation condition can inject a real numerics
    change (fp16 in place of LAW-06's pinned bf16). Every gating row calls it at the
    default; nothing in the production path reads it.
    """
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
    """PREREG_DFIX §1's logit statistic: |delta| / (|a| + 1e-3). The floor keeps a logit
    that happens to sit near zero from manufacturing an unbounded ratio — and IS the
    mechanism that destroyed the MAX reduction of this same expression (R181)."""
    return (treat - ref).abs() / (ref.abs() + 1e-3)


def median_form(treat: torch.Tensor, ref: torch.Tensor) -> float:
    """THE RE-POINTED STATISTIC (R181): the MEDIAN reduction of `rel`.

    **ITS CUDA NULL IS DEVICE-SPECIFIC. Do not quote the next line without the device.**

    Measured null on CUDA, **RTX 5080 / sm_120 / torch 2.11.0+cu128 ONLY**: exactly
    0.000000e+00 on 3675/3675 identical-code pairs. **On an RTX 4060 / sm_89 /
    torch 2.11.0+cu130 the same statistic on the same fixture reads 0/15 pairs zero, worst
    1.395037e-02** — 14.0x the old envelope and above F1's own maximum effect. The zero is a
    property of the 5080's kernel majority-bit-identity, NOT of this statistic (R191).

    Measured null on CPU: bit-identity, 1140/1140 — that one IS device-independent in
    practice and is asserted as exact equality. Measured under F1 on the 5080: 7.4900e-3 …
    1.3651e-2 over 3000 pairs.

    **Consequence for callers: no CUDA null leg bounds this statistic any more.** Both null
    legs assert EXACT equality — CPU natively, CUDA under `deterministic_algorithms()`. Every
    figure above is re-derived from the pinned artifact by `test_bf16_parity_nulldist.py`,
    not asserted here.
    """
    return float(rel(treat, ref).median())
