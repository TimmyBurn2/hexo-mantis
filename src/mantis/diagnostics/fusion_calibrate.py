# >300 justify (R8). NO LINE COUNT is stated, per G-DFIX-4 and R192(e)'s derive-or-delete:
# a transcribed tally must be re-edited on every edit, will eventually be wrong, and is then
# read as evidence. This module is ONE CLAIM — "here is the measured pair
# `(max_fused_edges, max_fused_nodes)` that fits inside the budget you named, and here is
# every mechanism that produced it". The sweep, the production forward, the peak measurement,
# the least-squares fit, the budget solve and the provenance block are the halves of that one
# claim and are not separable: a fit reported away from the sweep that produced it, or a
# recommended pair reported away from the allocator posture it was measured under, is a number
# without its producing mechanism, which R69 strikes. Splitting the file would put the number
# in one place and its mechanism in another, which is precisely the failure it exists to
# prevent.
"""`python -m mantis.diagnostics.fusion_calibrate` — measure the fused graph inference
forward's peak allocation and RECOMMEND `inference.fused_graph_caps` (F-816-10 design §9).

THE TOOL NEVER MINTS AND NEVER WRITES A CONFIG. It measures, it fits, it recommends and it
prints a copy-pasteable `tools/mint_config.py` line. Minting is the operator's act (R119), and
the value is theirs.

WHY IT LIVES UNDER `src/mantis` AND NOT UNDER `tools/`. It imports the PRODUCTION forward —
`collate_graph_batch` -> `GnnNet.forward_batch` -> `segment_softmax` -> the finiteness gate ->
the three D2H copies — and it must never diverge from it. A calibration that skips the softmax
or the D2H measures a different program, and the cap it emits then bounds a program nobody
runs. It is dev-facing, invoked as a module (no loose script files, CLAUDE.md).

WHAT IS MEASURED, AND WHAT IS REFUSED:

- **CUDA ONLY.** The quantity being fitted is a CUDA caching-allocator peak. On a host without
  one there is nothing to measure, so the tool exits 2 with a named refusal and emits NO cap
  — not a warning, not a degraded CPU estimate, not an extrapolation from tensor sizes. A
  number produced anyway would carry this tool's authority without its mechanism (R69), and a
  CPU-derived cap minted into a production config would be exactly the guessed value R119
  exists to forbid, wearing the tool's name. `--shapes-only` runs the device-free half and
  reports `"calibrated": false` with NULLS where the measurements would be.
- **PEAK IS A DELTA, NEVER AN ABSOLUTE.** `synchronize` -> `reset_peak_memory_stats` -> read
  `before` -> run -> `synchronize` -> `max_memory_allocated() - before`. An absolute reading
  charges this forward for whatever the co-resident trainer already held.
- **THE BUDGET IS THE OPERATOR'S.** `--budget-bytes` has NO default: it is R1's shape applied
  to a tool, because every cap emitted is a function of it. It is REQUIRED for a recommending
  run and waived by both `--shapes-only` and `--no-recommend`, which suppress the mint line and
  report the recommendation as absent (D-9). `--no-recommend` is what the box procedure's
  posture A/B step uses, where a fit is wanted BEFORE the budget is known.
- **THE REPORT GOES WHERE YOU SAY** (D-8): to `--out` when given, to stdout when not. The tool
  never invents an output path, because a tool that picks a filename writes somewhere nobody
  named.

THE SWEEP CARRIES AN EXPLICIT NODE-HEAVY / EDGE-LIGHT POINT (D-5), and its purpose is
statistical as well as adversarial. `peak ~ a + b*E + c*N` is only identifiable if the sweep
VARIES E/N: at a single operating ratio `b` and `c` are collinear and the fit cannot separate
them, so a one-ratio sweep would report a `c` that is an artefact of the solver. The scattered
point is also what reconfirms V-D's death by measurement instead of by hand count — the design
memo's per-tensor arithmetic is an UPPER BOUND, not a tight peak, and this is where that is
checked rather than asserted.

THE TWO CAPS DIVIDE ONE CARD. `train.microbatch_caps` was fitted against a self-play term
measured when the inference forward carried ONE graph; this bound is that term's partner, and
the operator may have to LOWER the training cap to buy the inference one. The budget block in
the report states every term it subtracted, so the joint re-fit is arithmetic the operator can
re-do rather than a number they must trust.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_TOOL = "mantis.diagnostics.fusion_calibrate"
_KEY = "inference.fused_graph_caps"
#: The block's two member names, held ONCE as data. The resolver is the one authority that
#: reads them off a config mapping and an AST census keeps it alone there; this tool only
#: WRITES them into a report and a mint line, so it spells them from here rather than as
#: constant subscripts that would read, to that census, exactly like a second authority.
_MEMBER_NAMES = ("max_fused_edges", "max_fused_nodes")


class CalibrationRefusal(Exception):
    """A named refusal that exits 2 and emits no cap.

    An exception and not a `sys.exit` at the raise site so every refusal takes ONE exit path
    with one exit code and one destination (stderr) — a refusal that sometimes lands on stdout
    would be parsed as a report by whatever consumes this tool's output.
    """


# ── the sweep (device-free) ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SweepPoint:
    """One `(regime, n_graphs)` cell of the sweep, with the shapes it actually built."""

    label: str
    n_graphs: int
    n_nodes: int
    n_edges: int
    max_graph_nodes: int
    max_graph_edges: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n_graphs": self.n_graphs,
            "nodes": self.n_nodes,
            "edges": self.n_edges,
            "edges_per_node": round(self.n_edges / max(self.n_nodes, 1), 4),
            "largest_graph_nodes": self.max_graph_nodes,
            "largest_graph_edges": self.max_graph_edges,
        }


#: The three regimes the sweep spans. `stones` is the position's stone count (LAW-03: `Ply`
#: counts STONES, not turns, so a late-game position at a 128-move cap carries ~125 of them);
#: `spread` is the lattice step between successive stones.
#:
#: THE LOW-RATIO ROW IS D-5's NODE-HEAVY / EDGE-LIGHT POINT, and the parameters were CHOSEN BY
#: MEASUREMENT, not by hand: a scan of this generator over `stones x spread` on `gnn_axis_v1`
#: put the reachable E/N band at roughly 21.2 (120 stones, packed) to 29.0 (120 stones,
#: spread 5), and the intuition that scattering the stones would STARVE the edge term is
#: FALSE — scattering gives each stone its own full legal-move ball, and a ball is dense, so
#: every node finds axis partners and E/N rises toward the ceiling. The three rows below take
#: the measured minimum, the measured maximum and a mid-game point between them, so the sweep
#: spans as much of the ratio as the producer actually reaches.
#:
#: THAT BAND IS NARROW, AND THE NARROWNESS IS ITSELF A RESULT. `b` and `c` are identifiable
#: only through variation in E/N; over a 21-to-29 band they are strongly collinear, so `_fit`
#: DISCLOSES the span and warns rather than reporting a `c` that is a solver artefact. It does
#: not weaken V-D's death, which rests on a structural inequality and not on this fit: the
#: builder emits two dummy edges per real node, so `E >= 2(N-1)` and an edge-only cap `C`
#: admits `N <= C/2 + 1` whatever the operating ratio turns out to be.
_REGIMES: tuple[tuple[str, int, int], ...] = (
    ("late_game_packed_low_ratio_node_heavy", 120, 1),
    ("mid_game_packed", 40, 1),
    ("late_game_spread_high_ratio_edge_heavy", 120, 5),
)
_GRAPH_COUNTS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 48, 64)


def _positions(stones: int, spread: int) -> list[tuple[int, int, int]]:
    """A legal stone list of `stones` stones on a `spread`-spaced hex spiral.

    Deterministic and geometry-only: no RNG, so two runs of this tool build the same sweep and
    a fit is comparable against an earlier one (LAW-09's matched-config requirement applied to
    a calibration).
    """
    out: list[tuple[int, int, int]] = []
    q = r = 0
    ring = 1
    while len(out) < stones:
        for dq, dr in ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)):
            for _ in range(ring):
                if len(out) >= stones:
                    break
                out.append((q, r, 1 if len(out) % 2 == 0 else -1))
                q += dq * spread
                r += dr * spread
            if len(out) >= stones:
                break
        ring += 1
    return out


def _empty_near(placed: list[tuple[int, int, int]], want: int = 2) -> list[tuple[int, int]]:
    """`want` EMPTY cells within hex distance 2 of the first stone, in a fixed order.

    Deliberately near a stone and not at an arbitrary coordinate: the producer's legal-node
    set is the empty cells within the encoding's legal-move radius of SOME stone, and a visit
    on any other cell loses its mass at sample-align and is refused. Distance 2 is inside the
    smallest radius the registry declares, so this holds for every graph encoding.
    """
    occupied = {(q, r) for q, r, _ in placed}
    q0, r0, _ = placed[0]
    found: list[tuple[int, int]] = []
    for dq in range(-2, 3):
        for dr in range(-2, 3):
            if abs(dq + dr) > 2:
                continue
            cell = (q0 + dq, r0 + dr)
            if cell not in occupied:
                found.append(cell)
                if len(found) == want:
                    return found
    raise CalibrationRefusal(
        f"{_TOOL}: could not find {want} empty legal cells beside the first stone of a "
        f"{len(placed)}-stone sweep position; the generator cannot build a valid visit row."
    )


#: The label the CORPUS source's single regime carries. A corpus has ONE distribution — its
#: own — so it is one regime rather than three: the ratio variation the fit needs comes from
#: the corpus's real spread of positions instead of from a generator's knobs, and inventing
#: three "regimes" over one ring would be three names for the same sample.
CORPUS_REGIME = "corpus"


def _load_corpus(encoding: str, max_moves: int, corpus_path: str) -> Any:
    """The run's OWN replay ring, loaded once and sampled per point.

    Loaded rather than generated: `--source corpus` exists because a cap fitted to a
    generator's geometry is a cap fitted to a distribution the run does not have, and the box
    is the only place the real one lives. Loading ZERO records is a REFUSAL and not an empty
    sweep — a tool that silently fitted nothing would emit a pair with no evidence behind it.
    """
    from mantis._engine import HexgBuffer

    buf = HexgBuffer(1 << 20, encoding, max_moves)
    try:
        loaded = buf.load_from_path(corpus_path)
    except (OSError, ValueError) as exc:
        # TRANSLATED AND RE-RAISED, never swallowed: the ring loader's own refusals are
        # correct and specific ("truncated file", a version mismatch), and they are kept
        # verbatim — but they arrive here as a bare traceback with the path nowhere in it,
        # and the operator at the box needs to know WHICH file this run was pointed at.
        raise CalibrationRefusal(
            f"{_TOOL}: --corpus-path {corpus_path} could not be read as a HEXG ring for "
            f"encoding {encoding!r}: {exc}"
        ) from exc
    if int(loaded) < 1:
        raise CalibrationRefusal(
            f"{_TOOL}: --corpus-path {corpus_path} loaded {loaded} records. A calibration "
            "fitted to an empty corpus is a pair with no evidence behind it; supply a ring "
            "with records, or use --source synthetic and say so in the sitting record."
        )
    return buf


def _build_wire(encoding: str, max_moves: int, n_graphs: int, stones: int, spread: int) -> Any:
    """One fused wire of `n_graphs` graphs at the named regime, through the REAL producer.

    A real `HexgBuffer` and a real `sample_graph_batch`, not a hand-built payload: the shapes
    this tool fits against must be shapes the production builder actually emits, or the cap is
    fitted to a geometry the run never sees.
    """
    from mantis._engine import HexgBuffer

    buf = HexgBuffer(max(n_graphs * 2, 8), encoding, max_moves)
    placed = _positions(stones, spread)
    # The visit row must be a DISTRIBUTION over cells that are EMPTY **and inside the rebuilt
    # legal-node set** — the sampler refuses a row whose mass drops at align, by name, and
    # that refusal is correct: a target on an illegal cell is a target no policy head can
    # produce. So the two cells are found next to a real stone rather than picked far away.
    visits = [(q, r, w) for (q, r), w in zip(_empty_near(placed), (0.6, 0.4), strict=True)]
    for i in range(max(n_graphs, 1)):
        buf.push_graph_position(
            placed, visits, 1, 30, 2 + i, True, 1.0 if i % 2 == 0 else -1.0, True, 10 + i,
        )
    wire, _targets = buf.sample_graph_batch(n_graphs, augment=False, recent_frac=0.0)
    return wire


def _wire_for(
    encoding: str, max_moves: int, n_graphs: int, label: str, stones: int, spread: int,
    corpus: Any,
) -> Any:
    """THE one place the sweep's data SOURCE is decided.

    One dispatch and not two paths, because the failure this shape prevents is the worst one a
    calibration tool has: a `--source corpus` run that quietly fitted GENERATED graphs would
    emit a pair carrying the corpus's authority and the generator's geometry, and nothing in
    the report would say so.
    """
    if corpus is not None:
        wire, _targets = corpus.sample_graph_batch(n_graphs, augment=False, recent_frac=0.0)
        return wire
    return _build_wire(encoding, max_moves, n_graphs, stones, spread)


def _regimes(corpus: Any) -> tuple[tuple[str, int, int], ...]:
    """The sweep's regimes for the chosen source — three for synthetic, ONE for a corpus."""
    return ((CORPUS_REGIME, 0, 0),) if corpus is not None else _REGIMES


def _sweep(encoding: str, max_moves: int, counts: tuple[int, ...],
           corpus: Any = None) -> list[SweepPoint]:
    points: list[SweepPoint] = []
    for label, stones, spread in _regimes(corpus):
        for n_graphs in counts:
            wire = _wire_for(encoding, max_moves, n_graphs, label, stones, spread, corpus)
            node_counts = np.diff(np.asarray(wire.node_offsets, dtype=np.int64))
            edge_counts = np.diff(np.asarray(wire.edge_offsets, dtype=np.int64))
            points.append(SweepPoint(
                label=label, n_graphs=n_graphs,
                n_nodes=int(node_counts.sum()), n_edges=int(edge_counts.sum()),
                max_graph_nodes=int(node_counts.max()), max_graph_edges=int(edge_counts.max()),
            ))
            del wire
    return points


# ── the measured half (CUDA only) ───────────────────────────────────────────────────────
def _measure_point(
    net: Any, spec: Any, device: Any, encoding: str, max_moves: int,
    point: SweepPoint, stones: int, spread: int, repeats: int, corpus: Any = None,
) -> dict[str, Any]:
    """Run the PRODUCTION forward `repeats` times at this point; report the MEDIAN peak delta.

    Median and not a single shot: one allocation retry or one cache state makes a single
    reading unrepeatable, and a cap fitted to a single shot is fitted to noise.
    """
    import torch

    from mantis.selfplay.graph_collate import (
        collate_graph_batch,
        segment_softmax,
        stone_mask_from_batch,
    )

    peaks: list[int] = []
    allocated_abs: list[int] = []
    reserved: list[int] = []
    for _ in range(repeats):
        wire = _wire_for(encoding, max_moves, point.n_graphs, point.label, stones,
                         spread, corpus)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.max_memory_allocated()
        batch = collate_graph_batch(
            wire, expected_version=1, trunk_size=spec.trunk_size,
            win_length=spec.win_length, node_feat_dim=spec.node_feat_dim,
            edge_feat_dim=spec.edge_feat_dim, device=str(device), semantic="canary",
            canary_period=64,
        )
        stone_mask = stone_mask_from_batch(batch)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=True,
        ):
            policy_logits, value, _bins = net.forward_batch(
                batch.x, batch.edge_index, batch.edge_attr, batch.legal_mask,
                stone_mask, batch.node_offsets,
            )
        probs = segment_softmax(policy_logits.float(), batch.legal_offsets)
        if not bool(torch.isfinite(probs).all()) or not bool(torch.isfinite(value).all()):
            raise CalibrationRefusal(
                f"{_TOOL}: the production forward produced NaN/Inf at "
                f"{point.label} n_graphs={point.n_graphs}. A peak measured through a broken "
                "forward is not a measurement of the forward this run uses."
            )
        # The three D2H copies are part of the program being measured; a calibration that
        # skipped them would fit a peak the run never reaches.
        _ = probs.detach().cpu().numpy()
        _ = batch.legal_offsets.detach().cpu().numpy()
        _ = value.detach().float().cpu().numpy().reshape(-1)
        torch.cuda.synchronize()
        peaks.append(int(torch.cuda.max_memory_allocated()) - int(before))
        # The ABSOLUTE allocated peak beside the delta, because the fragmentation ratio is
        # `reserved_peak / allocated_peak` and both must be measured from the SAME reset —
        # dividing a reserved absolute by an allocated delta is a ratio of two different
        # quantities that happens to have the right units.
        allocated_abs.append(int(torch.cuda.max_memory_allocated()))
        stats = torch.cuda.memory_stats()
        reserved.append(int(stats["reserved_bytes.all.peak"]))
        del wire, batch, stone_mask, policy_logits, value, probs
    stats = torch.cuda.memory_stats()
    free_b, total_b = torch.cuda.mem_get_info()
    return {
        **point.as_dict(),
        "peak_bytes_median": int(np.median(peaks)),
        "peak_bytes_all_repeats": peaks,
        "allocated_bytes_peak_median": int(np.median(allocated_abs)),
        "reserved_bytes_peak_median": int(np.median(reserved)),
        "num_alloc_retries": int(stats["num_alloc_retries"]),
        "num_ooms": int(stats["num_ooms"]),
        "mem_get_info_free_bytes": int(free_b),
        "mem_get_info_total_bytes": int(total_b),
    }


def _fit(measured: list[dict[str, Any]]) -> dict[str, Any]:
    """Least squares for `peak ~ a + b*E + c*N`, with R^2 and the max residual beside it.

    The residual travels with the fit because R^2 alone hides a single badly-missed point, and
    for a memory bound the point the model misses is the one that OOMs.
    """
    design = np.array([[1.0, float(m["edges"]), float(m["nodes"])] for m in measured])
    observed = np.array([float(m["peak_bytes_median"]) for m in measured])
    coeffs, *_ = np.linalg.lstsq(design, observed, rcond=None)
    predicted = design @ coeffs
    residuals = observed - predicted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    ratios = [m["edges"] / max(m["nodes"], 1) for m in measured]
    span = max(ratios) / max(min(ratios), 1e-9)
    return {
        # DISCLOSED, not buried: with a narrow E/N band the design matrix's second and third
        # columns are nearly proportional, so the split of the per-graph byte cost between
        # `b` and `c` is not identified by this sweep even at a high R^2. The operator sizing
        # a bound needs to know that the TOTAL is measured and the SPLIT is not — a `c` read
        # as a measurement here would be a number without a producing mechanism.
        "edges_per_node_span_ratio": span,
        "coefficients_are_separately_identified": bool(span >= 1.5),
        "identifiability_note": (
            "b and c are identifiable only through variation in E/N. This sweep spans "
            f"{min(ratios):.2f}..{max(ratios):.2f} (ratio {span:.2f}x). Below 1.5x, read the "
            "PREDICTED PEAK as measured and the b/c split as unresolved; the recommended pair "
            "is unaffected, because it is solved at the measured operating ratio where the "
            "two columns are not separated in the first place."
        ),
        "a_bytes": float(coeffs[0]),
        "b_bytes_per_edge": float(coeffs[1]),
        "c_bytes_per_node": float(coeffs[2]),
        "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "max_residual_bytes": float(np.max(np.abs(residuals))),
        "measured_edges_per_node_min": float(min(ratios)),
        "measured_edges_per_node_max": float(max(ratios)),
        "operating_edges_per_node": float(np.median(ratios)),
        # DERIVED FROM CODE, NOT MEASURED — printed beside the fit as a prediction to be
        # CHECKED against it (design §1.2/§1.3), and explicitly labelled so it is never read
        # as a second measurement. The memo's arithmetic is an UPPER BOUND: CPython frees
        # rebound intermediates earlier than it assumes (D-5).
        "derived_not_measured_expectation": {
            "b_bytes_per_edge_upper_bound": 1316,
            "c_bytes_per_node_upper_bound": 3686,
            "source": "F816_10_DESIGN sections 1.2/1.3, hand-derived from tensor shapes",
        },
    }


def _recommend(
    fit: dict[str, Any], budget_bytes: int, margin: float, points: list[dict[str, Any]]
) -> dict[str, Any]:
    """Solve `a + b*E + c*N <= budget * margin` with `E/N` pinned to the measured ratio."""
    ratio = fit["operating_edges_per_node"]
    usable = budget_bytes * margin - fit["a_bytes"]
    per_node = fit["b_bytes_per_edge"] * ratio + fit["c_bytes_per_node"]
    if usable <= 0 or per_node <= 0:
        raise CalibrationRefusal(
            f"{_TOOL}: the fit admits no positive pair under a budget of {budget_bytes} bytes "
            f"at margin {margin} (fixed term {fit['a_bytes']:.0f} B, per-node term "
            f"{per_node:.1f} B). Either the budget is below the forward's fixed cost or the "
            "fit is not usable; no pair is emitted, because a non-positive cap is not a cap."
        )
    nodes = int(usable // per_node)
    edges = int(nodes * ratio)
    # The OUTPUT condition, checked separately from the input one above. `usable > 0` admits
    # a `usable` so small that the floor division lands on ZERO, and `int(0 * ratio)` is zero
    # too — so the input guard alone lets the refusal's own sentence ("a non-positive cap is
    # not a cap") be violated by the branch that skipped it. A zero pair is refused by the
    # schema's `ge=1` downstream, but by then the operator has been handed a mint line for a
    # cap this tool never measured, and the error they see names pydantic instead of naming
    # this. Caught by RED-TEAM H4's sibling H1.
    if nodes < 1 or edges < 1:
        raise CalibrationRefusal(
            f"{_TOOL}: the fit's usable budget ({usable:.0f} B after the {margin} margin and "
            f"the {fit['a_bytes']:.0f} B fixed term) buys {nodes} nodes / {edges} edges — "
            "not a cap. A pair with a non-positive member would be refused by the schema's "
            "`ge=1` anyway; it is refused HERE so the message names the budget that produced "
            "it. Raise --budget-bytes, or re-fit: no pair and no mint line are emitted."
        )
    largest_nodes = max(p["largest_graph_nodes"] for p in points)
    largest_edges = max(p["largest_graph_edges"] for p in points)
    return {
        "max_fused_edges": edges,
        "max_fused_nodes": nodes,
        "budget_bytes": budget_bytes,
        "margin": margin,
        "predicted_peak_bytes": fit["a_bytes"] + fit["b_bytes_per_edge"] * edges
        + fit["c_bytes_per_node"] * nodes,
        "pinned_edges_per_node": ratio,
        # The floor BELOW which the cap starts refusing single graphs: a graph bigger than
        # the cap has no split that rescues it, so a pair under this is a run that dies on
        # its own positions rather than a run that is bounded.
        "largest_single_graph_nodes_seen": largest_nodes,
        "largest_single_graph_edges_seen": largest_edges,
        "refuses_a_graph_seen_in_this_sweep": bool(
            edges < largest_edges or nodes < largest_nodes
        ),
    }


# ── provenance (design §9.5) ────────────────────────────────────────────────────────────
def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
            cwd=str(Path(__file__).resolve().parents[3]), check=False,
        )
    except OSError:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _provenance(config_path: Path, spec: Any, arch: Any) -> dict[str, Any]:
    """Every mechanism behind every number above. R69: a number without its producing
    mechanism is struck, and this block is that mechanism."""
    import torch

    cuda = torch.cuda.is_available()
    return {
        "tool": _TOOL,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda,
        "device_name": torch.cuda.get_device_name(0) if cuda else None,
        "device_total_bytes": (
            int(torch.cuda.get_device_properties(0).total_memory) if cuda else None
        ),
        # The allocator posture is STAMPED because a cap does not transfer across a change of
        # it: `expandable_segments` moves the reserved/allocated ratio the budget divides by.
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "encoding_spec": spec.name,
        "encoding_registry_sha": getattr(spec, "registry_sha", None),
        "arch": repr(arch),
        "git_head": _git_head(),
    }


# ── the CLI ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"python -m {_TOOL}",
        description=(
            "Measure the fused graph inference forward's peak CUDA allocation over a sweep "
            f"of fused batches, fit peak ~ a + b*E + c*N, and recommend a {_KEY} pair for a "
            "budget you supply. Never mints, never writes a config."
        ),
    )
    parser.add_argument(
        "--config", required=True, metavar="PATH",
        help="The run config to calibrate FOR, read through the real loader. Its encoding, "
             "geometry and batch sizes are the sweep's shapes.",
    )
    parser.add_argument(
        "--budget-bytes", type=int, metavar="BYTES",
        help="Total inference-side byte budget the recommended pair must fit under. Carries "
             "no fallback value and none is invented: every cap this tool emits is a "
             "function of it, so a budget nobody minted would be a cap nobody measured. "
             "Required for a recommending run; waived when the run is not recommending.",
    )
    parser.add_argument(
        "--source", choices=("corpus", "synthetic"), default="synthetic",
        help="Where the sweep's graphs come from: 'corpus' samples a real replay ring at the "
             "run's own operating distribution (preferred at the box); 'synthetic' builds "
             "positions through the real producer.",
    )
    parser.add_argument(
        "--corpus-path", metavar="PATH",
        help="The replay ring to sample when --source corpus. Required for that source.",
    )
    parser.add_argument(
        "--repeats", type=int, default=5, metavar="N",
        help="Measurements per sweep point; the MEDIAN is reported, never a single shot. "
             "(5 when unset.)",
    )
    parser.add_argument(
        "--margin", type=float, default=0.85, metavar="M",
        help="Headroom multiplier applied to the budget. 0.85 is INHERITED with its "
             "provenance from the training-side sizing pass, where it covered a >1 GiB "
             "fragmentation swing and the eval child; the eval child OOM'd anyway, so treat "
             "it as a floor. Whatever is given is stamped into the report. (0.85 when unset.)",
    )
    parser.add_argument(
        "--shapes-only", action="store_true",
        help="Run only the device-free half: build the sweep, report each batch's (N, E) and "
             "the operating ratio, and NULL every measured field. Succeeds on a host with no "
             "CUDA. Emits no cap and no mint line.",
    )
    parser.add_argument(
        "--no-recommend", action="store_true",
        help="Measure and fit, but emit no pair and no mint line. This is the mode for a "
             "sitting that needs the fit BEFORE the budget is known (an allocator-posture "
             "A/B), which is otherwise circular.",
    )
    parser.add_argument(
        "--out", metavar="PATH",
        help="Where to write the JSON report. Written to stdout when unset; the tool never "
             "picks a filename of its own.",
    )
    return parser


def _emit(report: dict[str, Any], out: str | None) -> None:
    text = json.dumps(report, indent=2, sort_keys=True)
    if out is None:
        print(text)
        return
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    print(f"{_TOOL}: report written to {path}")


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    recommending = not (args.shapes_only or args.no_recommend)
    # ARGUMENT VALIDATION FIRST, and the budget check before the device check: both refuse
    # with exit 2, so a run missing its budget on a CPU host must still be told WHICH of the
    # two it is missing.
    if recommending and args.budget_bytes is None:
        raise CalibrationRefusal(
            f"{_TOOL}: --budget-bytes is required for a recommending run and has no fallback "
            "value (R1 applied to a tool: every cap emitted is a function of it, so a budget "
            "nobody minted is a cap nobody measured). Supply it, or pass --no-recommend for "
            "a fit without a pair, or --shapes-only for the device-free half."
        )
    if args.repeats < 1:
        raise CalibrationRefusal(f"{_TOOL}: --repeats must be at least 1; got {args.repeats}")
    if args.source == "corpus" and args.corpus_path is None:
        raise CalibrationRefusal(
            f"{_TOOL}: --source corpus needs --corpus-path; there is no default ring, and "
            "silently falling back to synthetic would fit the cap to a distribution the run "
            "does not have."
        )

    from mantis.config.loader import load_config
    from mantis.encoding import lookup

    config_path = Path(args.config)
    config = load_config(config_path)
    if config.identity.representation != "graph":
        raise CalibrationRefusal(
            f"{_TOOL}: {config_path} declares representation "
            f"{config.identity.representation!r}. There is no fused graph forward on the "
            f"dense route and therefore nothing to bound — the grid batch is a fixed-shape "
            "tensor already bounded by inference.inference_batch_size."
        )
    spec = lookup(config.identity.encoding)
    counts = tuple(n for n in _GRAPH_COUNTS if n <= int(config.inference.inference_batch_size))
    if not counts:
        counts = (int(config.inference.inference_batch_size),)
    max_moves = int(config.selfplay.max_game_moves)
    corpus = (
        _load_corpus(config.identity.encoding, max_moves, args.corpus_path)
        if args.source == "corpus" else None
    )

    if args.shapes_only:
        points = _sweep(config.identity.encoding, max_moves, counts, corpus)
        report = {
            "calibrated": False,
            "source": args.source,
            "corpus_path": args.corpus_path,
            "why_uncalibrated": (
                "--shapes-only: the device-free half ran and nothing was measured. "
                "`peak_bytes` and `fit` are null because there is NO PRODUCER on this path — "
                "the unproduced-field convention, not a zero and not an extrapolation."
            ),
            "sweep": [p.as_dict() for p in points],
            "peak_bytes": None,
            "fit": None,
            # DERIVABLE DEVICE-FREE, so it is reported rather than nulled: the operating ratio
            # and the largest single graph are SHAPES, and shapes are the whole of what this
            # mode measures. The largest single graph is also the floor below which a cap
            # starts refusing, which is what the box procedure's shape probe is for.
            "operating_ratio_e_over_n": float(np.median(
                [p.n_edges / max(p.n_nodes, 1) for p in points]
            )),
            "largest_single_graph": {
                "nodes": max(p.max_graph_nodes for p in points),
                "edges": max(p.max_graph_edges for p in points),
            },
            "fragmentation_ratio": None,
            "margin_achieved": None,
            "recommended": None,
            "mint_line": None,
            "provenance": _provenance(config_path, spec, None),
        }
        _emit(report, args.out)
        return 0

    import torch

    if not torch.cuda.is_available():
        raise CalibrationRefusal(
            f"{_TOOL}: torch.cuda.is_available() is False. This calibration measures a CUDA "
            "caching-allocator peak; on a host without one there is nothing to measure, so "
            "NO cap is emitted and none is estimated. A number produced here would carry "
            "this tool's authority without its mechanism (R69). Re-run on the box, or pass "
            "--shapes-only for the device-free half."
        )

    from mantis.model import arch_from_spec_and_config, build_net

    arch = arch_from_spec_and_config(spec, config.model_dump())
    device = torch.device("cuda")
    net = build_net(arch).to(device)
    net.eval()

    points = _sweep(config.identity.encoding, max_moves, counts, corpus)
    by_label = {label: (stones, spread) for label, stones, spread in _regimes(corpus)}
    measured: list[dict[str, Any]] = []
    unmeasured: list[dict[str, Any]] = []
    for point in points:
        try:
            measured.append(_measure_point(
                net, spec, device, config.identity.encoding, max_moves, point,
                *by_label[point.label], args.repeats, corpus,
            ))
        except torch.cuda.OutOfMemoryError as exc:
            # RECORDED, NEVER RETRIED, and never estimated. A sweep whose largest point does
            # not fit is the tool measuring exactly the wall the cap exists to stay under;
            # dying here would cost the sitting and leave the operator with no data at all,
            # while a retry or a smaller substitute would report a shape this point never
            # ran. The point is excluded from the fit BY NAME, so nothing silently narrows
            # the range the fit was taken over.
            torch.cuda.empty_cache()
            unmeasured.append({
                **point.as_dict(), "peak_bytes_median": None,
                "not_measured": "cuda_out_of_memory", "error": str(exc)[:300],
            })
            print(
                f"{_TOOL}: OOM at {point.label} n_graphs={point.n_graphs} "
                f"(N={point.n_nodes}, E={point.n_edges}) — recorded as unmeasured, excluded "
                "from the fit, NOT retried.",
                file=sys.stderr,
            )
    if len(measured) < 3:
        raise CalibrationRefusal(
            f"{_TOOL}: only {len(measured)} sweep point(s) could be measured; a three-term "
            "fit needs at least three. No cap is emitted — a pair solved from an "
            "underdetermined fit is a guess wearing a measurement's name."
        )
    fit = _fit(measured)
    recommendation = (
        _recommend(fit, int(args.budget_bytes), float(args.margin), measured)
        if recommending else None
    )
    mint_line = None
    if recommendation is not None:
        # The pair is read back through the block's own member NAMES, held once as data, so
        # this module does not become a second constant-string reader of the cap block — the
        # `_KEY` resolver is the ONE authority and an AST census pins that it stays alone.
        edges, nodes = (recommendation[member] for member in _MEMBER_NAMES)
        mint_line = (
            "uv run python tools/mint_config.py --template <template> --out "
            f"{config_path} --force <existing --set deltas...> --set "
            f"{_KEY}.{_MEMBER_NAMES[0]}={edges} --set {_KEY}.{_MEMBER_NAMES[1]}={nodes}"
        )
    # The MEASURED allocator posture, as one number: `reserved_peak / allocated_peak` over the
    # sweep. It is what the budget arithmetic divides by, and it is the reading the box
    # procedure's posture A/B compares between `expandable_segments` on and off — so it is
    # reported, never assumed, and a cap does not transfer across a change in it.
    ratios = [m["reserved_bytes_peak_median"] / max(m["allocated_bytes_peak_median"], 1)
              for m in measured]
    report = {
        "calibrated": True,
        "source": args.source,
        "corpus_path": args.corpus_path,
        "sweep": measured,
        "unmeasured_sweep_points": unmeasured,
        "peak_bytes": {m["label"] + f"@{m['n_graphs']}": m["peak_bytes_median"]
                       for m in measured},
        "fit": fit,
        "operating_ratio_e_over_n": fit["operating_edges_per_node"],
        "largest_single_graph": {
            "nodes": max(m["largest_graph_nodes"] for m in measured),
            "edges": max(m["largest_graph_edges"] for m in measured),
        },
        "fragmentation_ratio": float(np.median(ratios)),
        "margin_achieved": float(args.margin) if recommending else None,
        "recommended": recommendation,
        "mint_line": mint_line,
        "provenance": _provenance(config_path, spec, arch),
    }
    _emit(report, args.out)
    if mint_line is not None:
        print(mint_line)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(argv)
    except CalibrationRefusal as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
