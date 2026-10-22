// The two net backends, the on-device witness that admits one, and the selection between them (contract #13).
import type * as Ort from "onnxruntime-web";
import type { Batch, Engine as Wasm, web_batch } from "../pkg/mantis_player.js";

export const INT_INPUTS = ["src", "dst", "code", "legal", "node_graph", "legal_graph", "divisor", "real", "mean_mask",
  "mean_den"] as const;
export type IntInput = (typeof INT_INPUTS)[number];
export type Arrays = { x: Float32Array; feat: number } & Record<IntInput, Int32Array>;
export type NetOut = { logits: Float32Array; probs: Float32Array; value: Float32Array };
export type Line = { logit: number | null; prob: number; value: number; gap: number };
export type WitnessRow = {
  backend: string; logits: number; probs: number; value: number; argmax_agreed: number; argmax_compared: number;
  pass: boolean; leaves: number; ms: number; error?: string;
};
export type Manifest = {
  format: string; encoding: { name: string; registry_sha: string; node_feat_dim: number };
  deploy: { resolved: DeployBlock; player: Record<string, number | boolean> };
  files: Array<{ name: string; sha256: string; bytes: number; dtype?: string; passes_its_line?: boolean;
    tensors?: Array<{ name: string; rows: number; cols: number; offset: number }> }>;
  witness: { lines: Record<string, Line> }; inputs: string[]; outputs: string[];
  stamp: Record<string, unknown>;
};
export type DeployBlock = {
  search_kind: string; c_puct: number; fpu_reduction: number; quiescence_enabled: boolean; quiescence_blend_2: number;
  leaf_batch_size: number;
  tactics: null | { leaf_turns: number; leaf_nodes: number; root_turns: number; root_nodes: number;
    audit: null | { turns: number; nodes: number; k: number; m: number; total_nodes: number; mode: string } };
};
export type DeviceWitness = {
  positions: Array<[Array<[number, number, number]>, number, number]>;
  logits: number[][]; probs: number[][]; value: number[];
};

/** A net the engine's batches go through. */
export interface Backend {
  readonly kind: "webgpu" | "cpu";
  run(arrays: Arrays): Promise<NetOut>;
  dispose(): void;
}

/** A batch's arrays as typed arrays, copied out of wasm. */
export function arraysOf(batch: Batch, feat: number): Arrays {
  const ints = Object.fromEntries(INT_INPUTS.map((k) => [k, batch.i32(k)])) as Record<IntInput, Int32Array>;
  return { x: batch.x(), feat, ...ints };
}

/** The engine's own batch of `positions` (`[stones as (q, r, p), side, stones left]`), copied out as arrays. */
export function buildBatch(glue: { web_batch: typeof web_batch }, encoding: string, positions: DeviceWitness["positions"],
  feat: number): Arrays {
  const batch = glue.web_batch(encoding, Int32Array.from(positions.flatMap(([s]) => s.flat())),
    Uint32Array.from(positions.map(([s]) => s.length)), Int32Array.from(positions.map(([, side]) => side)),
    Int32Array.from(positions.map(([, , left]) => left)));
  const arrays = arraysOf(batch, feat);
  batch.free();
  return arrays;
}

/** The graphs of `a`, split at graph boundaries into calls of at most `cap` edges (a larger graph alone), re-based. */
export function splitBatch(a: Arrays, cap: number): Arrays[] {
  const graphs = a.mean_den.length;
  const nodeStart = new Int32Array(graphs + 1);
  const edgeStart = new Int32Array(graphs + 1);
  const legalStart = new Int32Array(graphs + 1);
  for (const g of a.node_graph) nodeStart[g + 1]! += 1;
  for (const g of a.legal_graph) legalStart[g + 1]! += 1;
  for (const d of a.dst) edgeStart[a.node_graph[d]! + 1]! += 1;
  for (let g = 0; g < graphs; g++) {
    nodeStart[g + 1]! += nodeStart[g]!;
    edgeStart[g + 1]! += edgeStart[g]!;
    legalStart[g + 1]! += legalStart[g]!;
  }
  const calls: Arrays[] = [];
  let lo = 0;
  while (lo < graphs) {
    let hi = lo + 1;
    while (hi < graphs && edgeStart[hi + 1]! - edgeStart[lo]! <= cap) hi++;
    const [n0, n1, e0, e1, l0, l1] = [nodeStart[lo]!, nodeStart[hi]!, edgeStart[lo]!, edgeStart[hi]!, legalStart[lo]!,
      legalStart[hi]!];
    const shift = (v: Int32Array, by: number) => v.map((x) => x - by);
    calls.push({
      feat: a.feat, x: a.x.slice(n0 * a.feat, n1 * a.feat),
      src: shift(a.src.slice(e0, e1), n0), dst: shift(a.dst.slice(e0, e1), n0), code: a.code.slice(e0, e1),
      legal: shift(a.legal.slice(l0, l1), n0), node_graph: shift(a.node_graph.slice(n0, n1), lo),
      legal_graph: shift(a.legal_graph.slice(l0, l1), lo), divisor: a.divisor.slice(n0, n1),
      real: a.real.slice(n0, n1), mean_mask: a.mean_mask.slice(n0, n1), mean_den: a.mean_den.slice(lo, hi),
    });
    lo = hi;
  }
  return calls;
}

function concat(parts: Float32Array[]): Float32Array {
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

/** What the WebGPU backend runs a call through: an ORT session, or a test's stub. */
export type Session = { run(feeds: Record<string, unknown>, fetches: string[]): Promise<Record<string, { data: unknown }>> };

/** ORT-web on WebGPU over the fp32 ONNX, a batch split into calls of at most `edgeCap` edges. */
export function ortBackend(ort: typeof Ort, session: Session, edgeCap: number): Backend {
  return {
    kind: "webgpu",
    async run(a) {
      const parts: NetOut[] = [];
      for (const call of splitBatch(a, edgeCap)) {
        const feeds: Record<string, unknown> = { x: new ort.Tensor("float32", call.x, [call.x.length / call.feat, call.feat]) };
        for (const k of INT_INPUTS) feeds[k] = new ort.Tensor("int32", call[k], [call[k].length]);
        const out = await session.run(feeds, ["logits", "probs", "value"]).catch((e: unknown) => {
          // ORT's own abort is a RuntimeError too; named so it is never read as a trap of the engine's module.
          throw Object.assign(new Error(e instanceof Error ? e.message : String(e)), { name: "BackendFailed" });
        });
        parts.push({ logits: out.logits!.data as Float32Array, probs: out.probs!.data as Float32Array,
          value: out.value!.data as Float32Array });
      }
      return { logits: concat(parts.map((p) => p.logits)), probs: concat(parts.map((p) => p.probs)),
        value: concat(parts.map((p) => p.value)) };
    },
    dispose() {
      (session as { release?: () => void }).release?.();
    },
  };
}

/** `RustForward` inside the engine's own module, on one CPU thread. */
export function cpuBackend(wasm: Wasm, mod: { Batch: typeof Batch }): Backend {
  return {
    kind: "cpu",
    async run(a) {
      const batch = new mod.Batch(a.x, a.src, a.dst, a.code, a.legal, a.node_graph, a.legal_graph, a.divisor, a.real,
        a.mean_mask, a.mean_den);
      const out = wasm.forward_batch(batch);
      batch.free();
      const net = { logits: out.logits(), probs: out.probs(), value: out.value() };
      out.free();
      return net;
    },
    dispose() {},
  };
}

/** Max |Δ| per output and the argmax agreed over the graphs whose reference top-2 gap exceeds the line's, for the
 * `graphs` reference rows from `from`; outputs of another length than the reference's fail every line. */
export function gaps(out: NetOut, ref: DeviceWitness, from: number, line: Line, graphs: number): Omit<WitnessRow, "backend" | "pass" | "leaves" | "ms"> {
  let at = 0;
  const row = { logits: 0, probs: 0, value: 0, argmax_agreed: 0, argmax_compared: 0 };
  const legal = ref.probs.slice(from, from + graphs).reduce((n, p) => n + p.length, 0);
  if (out.value.length !== graphs || out.probs.length !== legal || out.logits.length !== legal) {
    return { ...row, logits: Infinity, probs: Infinity, value: Infinity };
  }
  for (let g = 0; g < graphs; g++) {
    const want = { logits: ref.logits[from + g]!, probs: ref.probs[from + g]! };
    const n = want.probs.length;
    let top = 0, second = -Infinity, best = 0;
    for (let i = 0; i < n; i++) {
      row.logits = Math.max(row.logits, Math.abs(out.logits[at + i]! - want.logits[i]!));
      row.probs = Math.max(row.probs, Math.abs(out.probs[at + i]! - want.probs[i]!));
      if (want.logits[i]! > want.logits[top]!) top = i;
      if (out.logits[at + i]! > out.logits[at + best]!) best = i;
    }
    for (let i = 0; i < n; i++) if (i !== top) second = Math.max(second, want.logits[i]!);
    row.value = Math.max(row.value, Math.abs(out.value[g]! - ref.value[from + g]!));
    if (n === 1 || want.logits[top]! - second > line.gap) {
      row.argmax_compared += 1;
      row.argmax_agreed += Number(best === top);
    }
    at += n;
  }
  return row;
}

/** A row holds its line: every gated gap within it and every compared argmax agreed. */
export function passes(row: Omit<WitnessRow, "backend" | "pass" | "leaves" | "ms">, line: Line): boolean {
  return (line.logit === null || row.logits <= line.logit) && row.probs <= line.prob && row.value <= line.value
    && row.argmax_agreed === row.argmax_compared;
}

/** The device witness on `backend`: the engine builds the 16 positions' batches at the deploy batch size, the backend
 * runs them, and each graph is held to the export's reference at the fp32 line. */
export async function witness(backend: Backend, build: (positions: DeviceWitness["positions"]) => Arrays,
  ref: DeviceWitness, line: Line, batchSize: number, now: () => number): Promise<WitnessRow> {
  const rows = [];
  const started = now();
  for (let from = 0; from < ref.positions.length; from += batchSize) {
    const slice = ref.positions.slice(from, from + batchSize);
    const out = await backend.run(build(slice));
    rows.push(gaps(out, ref, from, line, slice.length));
  }
  const ms = now() - started;
  const row = combine(rows);
  // A witness that compared no argmax compared nothing: it does not pass.
  const pass = rows.length > 0 && row.argmax_compared > 0 && passes(row, line);
  return { backend: backend.kind, ...row, pass, leaves: ref.positions.length, ms };
}

/** The worst gap per output and the argmax counts summed, over a witness's batches. */
export function combine(rows: Array<Omit<WitnessRow, "backend" | "pass" | "leaves" | "ms">>): Omit<WitnessRow, "backend" | "pass" | "leaves" | "ms"> {
  return {
    logits: Math.max(0, ...rows.map((r) => r.logits)), probs: Math.max(0, ...rows.map((r) => r.probs)),
    value: Math.max(0, ...rows.map((r) => r.value)), argmax_agreed: rows.reduce((n, r) => n + r.argmax_agreed, 0),
    argmax_compared: rows.reduce((n, r) => n + r.argmax_compared, 0),
  };
}

/** The first candidate whose witness passes, in order; a candidate that throws is a failed row, except a refused net
 * file or a trap of the engine's own module, which end selection. Every row is kept for `NoBackendPassed`. */
export async function select(candidates: Array<[WitnessRow["backend"], () => Promise<Backend | null>]>,
  admit: (b: Backend) => Promise<WitnessRow>): Promise<{ backend: Backend; row: WitnessRow; rows: WitnessRow[] } | { rows: WitnessRow[] }> {
  const rows: WitnessRow[] = [];
  for (const [kind, make] of candidates) {
    let backend: Backend | null = null;
    try {
      backend = await make();
      if (backend === null) continue;
      const row = await admit(backend);
      rows.push(row);
      if (row.pass) return { backend, row, rows };
    } catch (e) {
      // A file that misses its manifest refuses the net, and a trap leaves the shared module unusable for any backend.
      if (e instanceof Error && (e.name === "Refused" || e instanceof WebAssembly.RuntimeError)) throw e;
      rows.push({ backend: kind, logits: Infinity, probs: Infinity, value: Infinity, argmax_agreed: 0, argmax_compared: 0,
        pass: false, leaves: 0, ms: 0, error: e instanceof Error ? `${e.name}: ${e.message}` : String(e) });
    }
    backend?.dispose();
  }
  return { rows };
}
