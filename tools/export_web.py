# >300 justify (R8): the detector, the re-expression, the export, the witness and the manifest are one pipeline over one
# stamp, kept in one file so the manifest a release ships cannot name a witness another file ran.
"""A stamp's browser net (contract #13), witnessed against the deploy path's served output: `uv run python tools/export_web.py --checkpoint STAMP --out DIR`."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxscript
import torch
from onnx import TensorProto, helper, numpy_helper
from torch import Tensor, nn

import mantis.selfplay.inference_server as server_mod
from mantis import _engine
from mantis.config.resolve.fused_graph_caps import resolve_fused_graph_caps
from mantis.config.resolve.inference_batching import resolve_inference_batching
from mantis.config.resolve.puct import resolve_puct_constants
from mantis.config.resolve.tactics import resolve_deploy_tactics
from mantis.encoding import handshake_ran, lookup, normalize_encoding_name
from mantis.model import build_net
from mantis.model.identity import net_param_hash
from mantis.selfplay.graph_collate import stone_mask_from_batch
from mantis.selfplay.inference_local import LocalInferenceEngine
from mantis.selfplay.served_graphs import SMALL_POP_FLOOR_NODES
from mantis.train.checkpoints import deploy_state, load_checkpoint
from mantis.util.git import head_sha, is_dirty

# ORT 1.31 reports to an outside endpoint at import unless this is set; the export and its gate send nothing.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
import onnxruntime as ort  # noqa: E402 — after the telemetry switch, which ORT reads at import

REPO = Path(__file__).resolve().parents[1]
GOLDENS = REPO / "tests" / "fixtures" / "graph_build" / "positions_r8.txt"
FORMAT = "mantis-web-manifest-v2"
BATCHES = (1, 8, 16)
INPUTS = ("x", "src", "dst", "code", "legal", "node_graph", "legal_graph", "divisor", "real", "mean_mask", "mean_den")
OUTPUTS = ("logits", "probs", "value")
#: The empty board and the first four stones: the goldens hold 4–171 stones, so the openings are witnessed beside them.
OPENINGS: tuple[tuple[list[tuple[int, int, int]], int, int], ...] = (
    ([], 1, 1),
    ([(0, 0, 1)], -1, 2),
    ([(0, 0, 1), (1, 0, -1)], -1, 1),
    ([(0, 0, 1), (1, 0, -1), (0, 1, -1)], 1, 2),
    ([(0, 0, 1), (1, 0, -1), (0, 1, -1), (-1, 0, 1)], 1, 1),
)
DEVICE_WITNESS_GOLDENS = 11
#: Every file an export writes, by kind, as the suffix after the stamp's stem; a refusal removes these and nothing else.
WRITTEN = {"fp32": ".fp32.onnx", "fp16": ".fp16.onnx", "weights": ".weights.bin", "witness": ".witness.json",
           "manifest": ".manifest.json"}
#: Ops that read an int32 index as well as an int64 one, by the input slot that carries it.
INDEX_SLOTS = {"Gather": 1, "GatherElements": 1, "ScatterElements": 1, "Unsqueeze": 0, "Expand": 0, "Reshape": 0}
#: The ops whose index input must reach the written graph as int32 wherever it is sized by the batch.
INDEX_READERS = {"Gather": 1, "GatherElements": 1, "ScatterElements": 1}

#: The detector's closed table: a carried value, or a refusal with its reason; anything else is not carried.
CARRIED_KINDS = ("GnnArchV2", "GnnArchV2SoftPolicy", "GnnArchV2SoftPolicyFuture")
REFUSED_KINDS = {"GnnArch": "its mean-only readout with no degree divisor is another graph, and no lineage net uses it"}
CARRIED_ENCODINGS = ("gnn_axis_r8", "gnn_axis_r8_pruned")
REFUSED_ENCODINGS = {"hexcanvas_r8": "the hex-canvas row has no model", "gnn_axis_v1": "radius 6 has no goldens"}
CARRIED_SEARCH = ("puct",)
REFUSED_SEARCH = {"gumbel": "the web player carries PUCT only"}

Case = tuple[dict[str, np.ndarray], dict[str, np.ndarray]]


@dataclass(frozen=True)
class Line:
    """A witness line: max |Δ| per output (None: reported, not gated) and the argmax rule's top-2 gap."""

    logit: float | None
    prob: float
    value: float
    gap: float


#: fp32: RESEARCH-WEB's pre-stated lines. fp16: R399(a), on what the search consumes, logits reported.
LINES = {"fp32": Line(logit=1e-3, prob=1e-4, value=1e-4, gap=2e-3), "fp16": Line(logit=None, prob=2e-3, value=2e-3, gap=2e-2)}


class ExportRefused(ValueError):
    """The stamp, the batch, the re-expression or the written graph is refused; `field` names what."""

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"export refused on {field}: {reason}")
        self.field = field


def detect(arch_kind: str, encoding: str, search_kind: str) -> None:
    """Refuse a stamp the exporter does not carry. Raises: ExportRefused — a refused or uncarried kind, row or search."""
    for field, value, carried, refused in (("arch_kind", arch_kind, CARRIED_KINDS, REFUSED_KINDS),
                                           ("encoding", encoding, CARRIED_ENCODINGS, REFUSED_ENCODINGS),
                                           ("deploy.search.kind", search_kind, CARRIED_SEARCH, REFUSED_SEARCH)):
        if value in carried:
            continue
        raise ExportRefused(field, refused.get(value, f"{value!r} is not carried"))


class WebServedGraph(nn.Module):
    """The served forward over contract #13's batch; every scatter-sum is ScatterElements, so no ScatterND is written."""

    def __init__(self, net: Any, vocab: Tensor) -> None:
        super().__init__()
        rep = net.representation
        self.input_proj, self.norms, self.final_norm = rep.input_proj, rep.norms, rep.final_norm
        self.mlps: Any = nn.ModuleList([c.nn for c in rep.convs])
        with torch.no_grad():
            pad = vocab.new_zeros(max(0, 1024 - vocab.shape[0]), vocab.shape[1])  # gine's table rows, as served
            proj = rep.edge_proj(torch.cat((vocab, pad)))
            for i, c in enumerate(rep.convs):
                self.register_buffer(f"table{i}", c.lin(proj)[: vocab.shape[0]].clone())
                self.register_buffer(f"eps{i}", c.eps.clone())
        self.policy = net.policy_head.mlp
        self.fc1, self.fc2 = net.value_head.fc1, net.value_head.fc2_bins
        self.register_buffer("support", torch.linspace(-1.0, 1.0, net.value_head.fc2_bins.out_features))

    @staticmethod
    def _sum(out: Tensor, index: Tensor, src: Tensor) -> Tensor:
        idx = index.long()  # torch's scatter wants int64; `narrow_index_casts` hands the written graph its int32 back
        return out.scatter_add(0, idx if src.dim() == 1 else idx[:, None].expand(-1, src.shape[1]), src)

    def forward(self, x: Tensor, src: Tensor, dst: Tensor, code: Tensor, legal: Tensor, node_graph: Tensor,
                legal_graph: Tensor, divisor: Tensor, real: Tensor, mean_mask: Tensor,
                mean_den: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        div = divisor.to(x.dtype).unsqueeze(-1)
        h = self.input_proj(x)
        hs = []
        for i, mlp in enumerate(self.mlps):
            hn = self.norms[i](h)
            msg = torch.relu(hn.index_select(0, src) + self.get_buffer(f"table{i}").index_select(0, code))
            agg = self._sum(torch.zeros_like(hn), dst, msg) / div
            h = torch.relu(mlp(agg + (1.0 + self.get_buffer(f"eps{i}")) * hn) + h)
            hs.append(self.final_norm(h))
        emb = torch.cat(hs, dim=-1)
        logits = self.policy(emb.index_select(0, legal)).squeeze(-1)
        g, d = mean_den.shape[0], emb.shape[1]
        mean = self._sum(emb.new_zeros(g, d), node_graph, emb * mean_mask.to(emb.dtype)[:, None])
        mean = mean / mean_den.to(emb.dtype)[:, None]
        floor = torch.finfo(emb.dtype).min
        masked = torch.where((real != 0)[:, None], emb, torch.full_like(emb, floor))
        maxed = emb.new_full((g, d), floor).scatter_reduce(0, node_graph.long()[:, None].expand(-1, d), masked, "amax")
        bins = self.fc2(torch.relu(self.fc1(torch.cat((mean, maxed), dim=-1))))
        value = (torch.softmax(bins, dim=-1) * self.get_buffer("support")).sum(-1).clamp(-1.0, 1.0)
        seg_max = logits.new_full((g,), float("-inf")).scatter_reduce(0, legal_graph.long(), logits, "amax")
        ex = torch.exp(logits - seg_max.index_select(0, legal_graph))
        probs = ex / self._sum(logits.new_zeros(g), legal_graph, ex).index_select(0, legal_graph)
        return logits, probs, value


def golden_positions(n: int) -> list[tuple[list[tuple[int, int, int]], int, int]]:
    """The first `n` golden positions as served requests: (stones, side to move, stones left this turn)."""
    out = []
    for line in GOLDENS.read_text(encoding="utf-8").splitlines():
        head, *fields = line.split()
        if head != "S":
            continue
        v = [int(f) for f in fields]
        out.append(([tuple(v[3 + 3 * i: 6 + 3 * i]) for i in range(v[2])], v[0], v[1]))
        if len(out) == n:
            break
    return out


def web_inputs(positions: list, encoding: str, node_feat_dim: int) -> dict[str, np.ndarray]:
    """Contract #13's batch for `positions`, built by the engine, `x` as `[N, node_feat_dim]`. Raises: ValueError."""
    batch = _engine.web_batch(positions, encoding)
    arrays = {k: np.asarray(batch[k]) for k in INPUTS}
    arrays["x"] = arrays["x"].reshape(-1, node_feat_dim)
    return arrays


def served_reference(net: nn.Module, spec: Any, config: dict, positions: list, batch: int) -> list[dict[str, np.ndarray]]:
    """The deploy path's served x, probs, value and eager logits for `positions` in groups of `batch`; Raises: RuntimeError — a group was not served as one batch; ValueError — a seam guard."""
    seen: list[dict[str, np.ndarray]] = []
    served = server_mod.served_outputs

    def record(model: Any, b: Any, edge_vocab: Tensor, *, trunk: Any, amp_dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        probs, value = served(model, b, edge_vocab, trunk=trunk, amp_dtype=amp_dtype)
        logits, _, _ = model.forward_batch(b.x, b.edge_index, b.edge_code, b.legal_node_gather,
                                           stone_mask_from_batch(b), b.node_offsets, edge_vocab=edge_vocab)
        seen.append({"x": b.x.detach().cpu().numpy().copy(), "logits": logits.float().numpy().copy(),
                     "probs": probs.numpy().copy(), "value": value.numpy().copy()})
        return probs, value

    server_mod.served_outputs = record
    try:
        engine = LocalInferenceEngine(
            net, torch.device("cpu"), encoding_spec=spec, fused_graph_caps=resolve_fused_graph_caps(config),
            inference_batching=resolve_inference_batching(config),
            max_in_flight=int(config["selfplay"]["leaf_batch_size"]), submitters=1, leaf_build_threads=1,
            bucket_floor_nodes=SMALL_POP_FLOOR_NODES)
        try:
            for i in range(0, len(positions), batch):
                before = len(seen)
                engine.infer_positions_ls(positions[i:i + batch])
                if len(seen) != before + 1:
                    raise RuntimeError(f"a group of {batch} was served as {len(seen) - before} batches")
        finally:
            engine.close()
    finally:
        server_mod.served_outputs = served
    return seen


def witness_cases(net: nn.Module, spec: Any, config: dict, encoding: str, positions: list,
                  batches: tuple[int, ...]) -> dict[int, list[Case]]:
    """Per batch size: the engine's batch beside the served reference, in the same order; Raises: ExportRefused — the engine's `x` is not the served one; RuntimeError, ValueError — as `served_reference`."""
    feat = int(spec.node_feat_dim)
    cases: dict[int, list[Case]] = {}
    with torch.inference_mode():
        for b in batches:
            refs = served_reference(net, spec, config, positions, b)
            cases[b] = [(web_inputs(positions[i:i + b], encoding, feat), ref)
                        for i, ref in zip(range(0, len(positions), b), refs, strict=True)]
    for b, pairs in cases.items():
        for inputs, ref in pairs:
            if not np.array_equal(inputs["x"], ref["x"]):
                raise ExportRefused("batch", f"the engine's batch at B {b} is not the served one")
    return cases


def gaps(out: dict[str, np.ndarray], ref: dict[str, np.ndarray], legal_graph: np.ndarray, line: Line) -> dict[str, Any]:
    """Max |Δ| per output, and the argmax agreed and compared over graphs whose reference top-2 gap exceeds the line's."""
    row: dict[str, Any] = {k: float(np.abs(out[k].astype(np.float64) - ref[k]).max()) for k in OUTPUTS}
    bounds = np.flatnonzero(np.diff(legal_graph)) + 1
    agreed = compared = 0
    for r, o in zip(np.split(ref["logits"], bounds), np.split(out["logits"], bounds), strict=True):
        top = np.sort(r)[::-1]
        if top.shape[0] > 1 and top[0] - top[1] <= line.gap:
            continue
        compared += 1
        agreed += int(np.argmax(r) == np.argmax(o))
    row["argmax_agreed"], row["argmax_compared"] = agreed, compared
    return row


def passes(row: dict[str, Any], line: Line) -> bool:
    """Whether a witness row holds its line."""
    return ((line.logit is None or row["logits"] <= line.logit) and row["probs"] <= line.prob
            and row["value"] <= line.value and row["argmax_agreed"] == row["argmax_compared"])


def ort_session(path: Path, threads: int) -> ort.InferenceSession:
    """ORT CPU on `path` at `threads` (0: ORT's default); Raises: onnxruntime's errors — a file ORT cannot load."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def rust_outputs(forward: Any, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """`RustForward`'s outputs on one case's arrays. Raises: ValueError — a batch the forward refuses."""
    out = forward.forward({**inputs, "x": np.ascontiguousarray(inputs["x"]).ravel()})
    return {k: np.asarray(out[k]) for k in OUTPUTS}


def rust_witness(weights: Path, table: list[dict[str, Any]], fp32: Path, cases: dict[int, list[Case]],
                 feat: int) -> list[dict[str, Any]]:
    """W2: `RustForward` over the weights file against the fp32 ONNX at ORT CPU 1 thread, one row per batch size. Raises: ValueError — a table the file does not hold or a refused batch."""
    forward = _engine.RustForward(table, np.fromfile(weights, dtype="<f4"), feat)
    session, line, rows = ort_session(fp32, 1), LINES["fp32"], []
    for b, pairs in cases.items():
        per = []
        for inputs, _ref in pairs:
            onnx_out = dict(zip(OUTPUTS, session.run(list(OUTPUTS), inputs), strict=True))
            per.append(gaps(rust_outputs(forward, inputs), onnx_out, inputs["legal_graph"], line))
        rows.append(witness_row(per, {"backend": "rust-forward-native", "file": weights.name, "threads": 1, "B": b}, "fp32"))
    return rows


def witness_row(per: list[dict[str, Any]], head: dict[str, Any], dtype: str) -> dict[str, Any]:
    """One witness row: `head`, the worst gap per output and the argmax counts over `per`, held to `dtype`'s line."""
    row = {**head, **{k: max(r[k] for r in per) for k in OUTPUTS},
           **{k: sum(r[k] for r in per) for k in ("argmax_agreed", "argmax_compared")}}
    row["pass"] = passes(row, LINES[dtype])
    row["line"] = dtype
    return row


def witness(path: Path, threads: int, cases: dict[int, list[Case]], dtype: str) -> list[dict[str, Any]]:
    """ORT CPU at `threads` (0: ORT's default) on `path`, one row per batch size, held to `dtype`'s line."""
    session = ort_session(path, threads)
    line, rows = LINES[dtype], []
    for b, pairs in cases.items():
        per = []
        for inputs, ref in pairs:
            got = {k: np.asarray(v) for k, v in zip(OUTPUTS, session.run(list(OUTPUTS), inputs), strict=True)}
            per.append(gaps(got, ref, inputs["legal_graph"], line))
        rows.append(witness_row(per, {"backend": "ort-cpu", "file": path.name, "threads": threads or "default", "B": b}, dtype))
    return rows


def export_onnx(wrapper: nn.Module, sample: dict[str, np.ndarray], path: Path, opset: int) -> None:
    """torch.onnx's dynamo export of `wrapper`, every axis dynamic, checked. Raises: RuntimeError — no program came back."""
    dim = torch.export.Dim
    nodes, edges, legal, graphs = dim("N", min=2), dim("E", min=1), dim("L", min=1), dim("G", min=1)
    axes = {"x": nodes, "src": edges, "dst": edges, "code": edges, "legal": legal, "node_graph": nodes,
            "legal_graph": legal, "divisor": nodes, "real": nodes, "mean_mask": nodes, "mean_den": graphs}
    with torch.no_grad():
        prog = torch.onnx.export(wrapper, tuple(torch.from_numpy(sample[k]) for k in INPUTS), dynamo=True,
                                 opset_version=opset, dynamic_shapes={k: {0: axes[k]} for k in INPUTS},
                                 input_names=list(INPUTS), output_names=list(OUTPUTS), optimize=True,
                                 external_data=False)
    if prog is None:
        raise RuntimeError("torch.onnx returned no program")
    prog.save(str(path))
    onnx.checker.check_model(onnx.load(str(path)))


def _value_types(model: onnx.ModelProto) -> dict[str, tuple[int, bool]]:
    """Every named value's inferred element type, and whether its shape is fully static (known, every dim a number)."""
    graph = onnx.shape_inference.infer_shapes(model).graph
    out = {}
    for v in [*graph.input, *graph.value_info, *graph.output]:
        tensor = v.type.tensor_type
        out[v.name] = (tensor.elem_type, tensor.HasField("shape") and all(d.HasField("dim_value") for d in tensor.shape.dim))
    for init in graph.initializer:
        out[init.name] = (init.data_type, True)
    return out


def narrow_index_casts(path: Path) -> None:
    """Drop every Cast of an int32 value to int64 whose every read takes either width (`INDEX_SLOTS`), wiring its readers to the int32 value; Raises: onnx.checker.ValidationError, onnx.shape_inference.InferenceError — the narrowed graph does not type-check."""
    model = onnx.load(str(path))
    graph, types = model.graph, _value_types(model)
    readers: dict[str, list[tuple[onnx.NodeProto, int]]] = {}
    for node in graph.node:
        for slot, name in enumerate(node.input):
            readers.setdefault(name, []).append((node, slot))
    dropped = []
    for node in graph.node:
        to = next((a.i for a in node.attribute if a.name == "to"), None)
        if node.op_type != "Cast" or to != TensorProto.INT64 or types.get(node.input[0], (0, False))[0] != TensorProto.INT32:
            continue
        reads = readers.get(node.output[0], [])
        if not reads or node.output[0] in {o.name for o in graph.output} or any(
                INDEX_SLOTS.get(u.op_type) != slot for u, slot in reads):
            continue
        for u, slot in reads:
            u.input[slot] = node.input[0]
        dropped.append(node)
    for node in dropped:
        graph.node.remove(node)
    del graph.value_info[:]  # the int64 types recorded on the rewired chains are stale; inference re-derives them
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(path))


def graph_refusal(ops: dict[str, int], model: onnx.ModelProto) -> str | None:
    """Why a written graph is refused: a ScatterND, a GatherND, or an index reader's index that is not int32 unless it is a fully static int64 (a shape op's); an untyped index refuses."""
    if "ScatterND" in ops:
        return "a ScatterND, which races under threaded ORT kernels"
    if "GatherND" in ops:
        return "a GatherND, whose int64 index the int32 batch was built to remove"
    types = _value_types(model)

    def int32_or_static(name: str) -> bool:
        elem, static = types.get(name, (TensorProto.UNDEFINED, False))
        return elem == TensorProto.INT32 or (elem == TensorProto.INT64 and static)

    wide = sorted({n.input[slot] for n in model.graph.node if (slot := INDEX_READERS.get(n.op_type)) is not None
                   and len(n.input) > slot and not int32_or_static(n.input[slot])})
    return f"indices not read as int32 {wide}" if wide else None


def fp16_storage(src: Path, dst: Path) -> None:
    """`src` with every float initializer that fp16 holds finite stored in fp16 and cast back to fp32 where read; compute stays fp32."""
    model = onnx.load(str(src))
    graph = model.graph
    keep, casts = [], []
    for init in graph.initializer:
        values = numpy_helper.to_array(init)
        with np.errstate(over="ignore"):
            half = values.astype(np.float16) if init.data_type == TensorProto.FLOAT else None
        if half is None or not np.array_equal(np.isfinite(half), np.isfinite(values)):
            keep.append(init)  # an int, or a constant like finfo(f32).min that fp16 would turn infinite
            continue
        stored = numpy_helper.from_array(half, init.name + "__fp16")
        keep.append(stored)
        casts.append(helper.make_node("Cast", [stored.name], [init.name], to=TensorProto.FLOAT))
    del graph.initializer[:]
    graph.initializer.extend(keep)
    nodes = casts + list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))


def op_census(path: Path) -> dict[str, int]:
    """Op counts of the written graph."""
    ops: dict[str, int] = {}
    for node in onnx.load(str(path)).graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    return dict(sorted(ops.items()))


def dump_weights(wrapper: WebServedGraph, path: Path) -> list[dict[str, Any]]:
    """The Rust forward's weights: raw f32 little-endian, linear weights transposed (in × out); returns the tensor table."""
    tensors: list[tuple[str, np.ndarray]] = []

    def linear(name: str, mod: nn.Linear) -> None:
        tensors.extend(((f"{name}.wt", mod.weight.detach().numpy().T), (f"{name}.b", mod.bias.detach().numpy())))

    def norm(name: str, mod: nn.LayerNorm) -> None:
        tensors.extend(((f"{name}.w", mod.weight.detach().numpy()), (f"{name}.b", mod.bias.detach().numpy())))

    linear("input_proj", wrapper.input_proj)
    for i, mlp in enumerate(wrapper.mlps):
        norm(f"norm{i}", wrapper.norms[i])
        tensors.extend(((f"table{i}", wrapper.get_buffer(f"table{i}").numpy()),
                        (f"eps{i}", wrapper.get_buffer(f"eps{i}").numpy())))
        linear(f"mlp{i}.0", mlp[0])
        linear(f"mlp{i}.2", mlp[2])
    norm("final_norm", wrapper.final_norm)
    linear("policy.0", wrapper.policy[0])
    linear("policy.2", wrapper.policy[2])
    linear("fc1", wrapper.fc1)
    linear("fc2", wrapper.fc2)
    tensors.append(("support", wrapper.get_buffer("support").numpy()))
    table, offset = [], 0
    with path.open("wb") as f:
        for name, a in tensors:
            a = np.ascontiguousarray(a, dtype="<f4")
            rows, cols = (1, a.size) if a.ndim <= 1 else a.shape
            table.append({"name": name, "rows": int(rows), "cols": int(cols), "offset": offset})
            f.write(a.tobytes())
            offset += a.size
    return table


def file_row(path: Path, **extra: Any) -> dict[str, Any]:
    """A shipped file's name, sha256 and bytes."""
    return {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
            **extra}


def device_witness(net: nn.Module, spec: Any, config: dict, encoding: str) -> dict[str, Any]:
    """The 16 positions a device runs at start (the openings and goldens spread over stone count) and their served outputs."""
    goldens = sorted(golden_positions(512), key=lambda p: len(p[0]))
    picks = [goldens[round(i * (len(goldens) - 1) / (DEVICE_WITNESS_GOLDENS - 1))] for i in range(DEVICE_WITNESS_GOLDENS)]
    positions = list(OPENINGS) + picks
    ((inputs, ref),) = witness_cases(net, spec, config, encoding, positions, (len(positions),))[len(positions)]
    bounds = np.flatnonzero(np.diff(inputs["legal_graph"])) + 1
    return {"positions": [[[list(s) for s in stones], side, left] for stones, side, left in positions],
            "logits": [a.tolist() for a in np.split(ref["logits"], bounds)],
            "probs": [a.tolist() for a in np.split(ref["probs"], bounds)],
            "value": ref["value"].tolist()}


def export(checkpoint: Path, out: Path, *, n_positions: int = 512, opset: int = 18) -> dict[str, Any]:
    """Detect, re-express, export, witness and write the manifest for `checkpoint` into `out`; Raises: ExportRefused — an uncarried stamp, an unchecked registry, a foreign batch, a re-expression not bit-identical, or a refused or untyped graph (its files removed); CheckpointStampError — the stamp itself; RuntimeError, ValueError, KeyError — a served group split, a seam guard, a config without its deploy or self-play block; onnx.checker.ValidationError — a written fp32 or fp16 file ONNX rejects; onnxruntime's errors — a session ORT cannot build or run."""
    # The tree that runs is the tree at the start: its commit and its tracked changes, read before any file is written.
    commit, dirty = head_sha(REPO), is_dirty(REPO) is not False
    ck = load_checkpoint(checkpoint, declared_encoding=None, declared_future_head=None)
    config = ck.config
    if not isinstance(config, dict):
        raise ExportRefused("config", "the stamp carries no config mapping")
    encoding = normalize_encoding_name(ck.metadata.encoding_name)
    arch_kind = type(ck.metadata.arch).__name__
    detect(arch_kind, encoding, str(config["deploy"]["search"]["kind"]))
    if not handshake_ran():
        raise ExportRefused("registry", "the on-disk registry was not checked against the compiled one")
    spec, arch = lookup(encoding), ck.metadata.arch
    if arch is None or spec.node_feat_dim is None or spec.edge_feat_dim is None or spec.win_length is None:
        raise ExportRefused("stamp", "no arch, or an encoding row without graph widths")
    net = build_net(arch)
    state, weights = deploy_state(ck)
    net.load_state_dict(state)
    net.eval()
    vocab = torch.from_numpy(np.asarray(_engine.edge_vocabulary(spec.win_length))).reshape(-1, spec.edge_feat_dim)
    wrapper = WebServedGraph(net, vocab).eval()

    positions = golden_positions(n_positions)
    cases = witness_cases(net, spec, config, encoding, positions, BATCHES)
    assert len(OPENINGS) not in BATCHES, "the openings' batch would overwrite a golden one"
    cases |= witness_cases(net, spec, config, encoding, list(OPENINGS), (len(OPENINGS),))
    with torch.inference_mode():
        for b, pairs in cases.items():
            for inputs, ref in pairs:
                got = dict(zip(OUTPUTS, wrapper(*(torch.from_numpy(inputs[k]) for k in INPUTS)), strict=True))
                worst = max(float(np.abs(got[k].numpy().astype(np.float64) - ref[k]).max()) for k in OUTPUTS)
                if worst != 0.0:
                    raise ExportRefused("re-expression", f"max |Δ| {worst:.3g} at B {b}; it must be bit-identical")

    out.mkdir(parents=True, exist_ok=True)
    stem = checkpoint.name.removesuffix(".ckpt")
    paths = {kind: out / f"{stem}{suffix}" for kind, suffix in WRITTEN.items()}
    for path in paths.values():
        path.unlink(missing_ok=True)  # an earlier export's files go first, so no manifest can name a file this one rewrote
    fp32, fp16 = paths["fp32"], paths["fp16"]
    t0 = time.perf_counter()
    export_onnx(wrapper, cases[8][0][0], fp32, opset)
    try:
        narrow_index_casts(fp32)
        refusal = graph_refusal(op_census(fp32), onnx.load(str(fp32)))
    except (onnx.checker.ValidationError, onnx.shape_inference.InferenceError) as exc:
        refusal = f"a narrowing that does not type-check ({exc})"
    export_s = time.perf_counter() - t0
    if refusal is not None:
        for path in paths.values():
            path.unlink(missing_ok=True)
        raise ExportRefused("graph", f"the written graph holds {refusal}")
    fp16_storage(fp32, fp16)
    weights_path = paths["weights"]
    table = dump_weights(wrapper, weights_path)

    rows = [*witness(fp32, 1, cases, "fp32"), *witness(fp32, 0, cases, "fp32"),
            *witness(fp16, 1, cases, "fp16"), *witness(fp16, 0, cases, "fp16"),
            *rust_witness(weights_path, table, fp32, cases, int(spec.node_feat_dim))]
    device = device_witness(net, spec, config, encoding)
    witness_path = paths["witness"]
    witness_path.write_text(json.dumps(device), encoding="utf-8")

    def verdict(path: Path) -> bool:
        return all(r["pass"] for r in rows if r["file"] == path.name)

    puct = resolve_puct_constants(config)
    manifest = {
        "format": FORMAT,
        "stamp": {"file": checkpoint.name, "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  "run_id": ck.metadata.run_id, "step": ck.metadata.step, "lineage": list(ck.metadata.lineage or []),
                  "commit_sha": ck.metadata.commit_sha, "arch_kind": arch_kind, "encoding": encoding,
                  "weights": weights, "net_param_hash": net_param_hash(net),
                  "arch": {k: v for k, v in asdict(arch).items() if k != "representation"}},
        "exporter": {"commit": commit, "dirty": dirty, "torch": torch.__version__,
                     "onnx": onnx.__version__, "onnxscript": onnxscript.__version__, "onnxruntime": ort.__version__,
                     "opset": opset, "export_s": round(export_s, 1)},
        "encoding": {"name": encoding, "registry_sha": _engine.registry_sha_hex(), "win_length": spec.win_length,
                     "graph_radius": spec.graph_radius, "trunk_size": spec.trunk_size,
                     "empty_edges": str(spec.empty_edges), "node_feat_dim": spec.node_feat_dim,
                     "edge_vocab_rows": int(vocab.shape[0])},
        "deploy": {"resolved": {"search_kind": config["deploy"]["search"]["kind"], **puct.tree_kwargs(),
                                "leaf_batch_size": int(config["selfplay"]["leaf_batch_size"]),
                                "tactics": resolve_deploy_tactics(config)},
                   "player": _engine.player_constants()},
        "inputs": list(INPUTS), "outputs": list(OUTPUTS),
        "files": [file_row(fp32, dtype="fp32", ops=op_census(fp32), passes_its_line=verdict(fp32)),
                  file_row(fp16, dtype="fp16", ops=op_census(fp16), passes_its_line=verdict(fp16)),
                  file_row(weights_path, dtype="fp32", tensors=table, passes_its_line=verdict(weights_path)),
                  file_row(witness_path, positions=len(device["positions"]))],
        "witness": {"lines": {k: asdict(v) for k, v in LINES.items()},
                    "goldens": {"path": str(GOLDENS.relative_to(REPO)), "sha256": hashlib.sha256(GOLDENS.read_bytes()).hexdigest(),
                                "positions": len(positions), "openings": len(OPENINGS)},
                    "rows": rows},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--positions", type=int, default=512)
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()
    torch.manual_seed(0)
    try:
        manifest = export(args.checkpoint, args.out, n_positions=args.positions, opset=args.opset)
    except ExportRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    failed = [r for r in manifest["witness"]["rows"] if r["line"] == "fp32" and not r["pass"]]
    for r in manifest["witness"]["rows"]:
        print(json.dumps(r))
    if failed:
        print(f"HALT: {len(failed)} fp32 witness row(s) miss their line", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
