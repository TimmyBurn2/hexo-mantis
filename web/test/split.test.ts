// The WebGPU backend's per-call edge cap: a batch split at graph boundaries gives the outputs of the whole batch, on a
// stub session whose outputs depend on each graph alone (so a re-basing error shows as a wrong value).
import assert from "node:assert/strict";
import { test } from "node:test";
import * as ort from "onnxruntime-web";
import { buildBatch, ortBackend, splitBatch, type Arrays, type Session } from "../src/backends.ts";
import { Glue, goldens } from "./host.ts";

const batch = (n: number): Arrays => buildBatch(Glue, "gnn_axis_r8", goldens(n), 11);

/** Each graph's outputs from its own rows of every array (a call's re-basing error moves them); a legal row whose graph
 * disagrees with its node's throws. */
const stub = (calls: number[]): Session => ({
  async run(feeds) {
    const f = Object.fromEntries(Object.entries(feeds).map(([k, t]) => [k, (t as ort.Tensor).data])) as Record<string, any>;
    calls.push(f.src.length);
    const graphs = f.mean_den.length;
    const value = new Float32Array(graphs);
    const degree = new Float32Array(f.node_graph.length);
    const feat = f.x.length / f.node_graph.length;
    for (let e = 0; e < f.src.length; e++) {
      value[f.node_graph[f.dst[e]]]! += f.code[e] * f.x[f.src[e] * feat];
      degree[f.dst[e]]! += 1;
    }
    for (let v = 0; v < f.node_graph.length; v++) {
      value[f.node_graph[v]]! += 0.5 * f.divisor[v] + 0.25 * f.real[v] + 0.125 * f.mean_mask[v];
    }
    for (let g = 0; g < graphs; g++) value[g]! += f.mean_den[g];
    for (let i = 0; i < f.legal.length; i++) {
      if (f.node_graph[f.legal[i]] !== f.legal_graph[i]) throw new Error(`legal row ${i} is in another graph than its node`);
    }
    const logits = Float32Array.from(f.legal, (v: number) => degree[v]!);
    return { logits: { data: logits }, probs: { data: logits }, value: { data: value } };
  },
});

test("a batch split under an edge cap answers what the whole batch answers, each call within the cap", async () => {
  const a = batch(16);
  const whole = await ortBackend(ort, stub([]), Number.MAX_SAFE_INTEGER).run(a);
  const calls: number[] = [];
  const cap = Math.ceil(a.src.length / 5);
  const split = await ortBackend(ort, stub(calls), cap).run(a);
  assert.ok(calls.length >= 5, `${calls.length} calls`);
  assert.ok(calls.every((e) => e <= cap || splitBatch(a, cap).some((c) => c.mean_den.length === 1 && c.src.length === e)));
  assert.deepEqual(split.value, whole.value);
  assert.deepEqual(split.logits, whole.logits);
});

test("a graph over the cap is a call of its own, never cut", () => {
  const a = batch(4);
  const calls = splitBatch(a, 1);
  assert.equal(calls.length, 4);
  assert.deepEqual(calls.map((c) => c.mean_den.length), [1, 1, 1, 1]);
  assert.equal(calls.reduce((n, c) => n + c.src.length, 0), a.src.length);
});

test("a witness fails a backend whose outputs are short, and one that compared no argmax", async () => {
  const { gaps, passes, witness } = await import("../src/backends.ts");
  const line = { logit: 1e-3, prob: 1e-4, value: 1e-4, gap: 2e-3 };
  const ref = { positions: [[[], 1, 1]] as never, logits: [[0, 1]], probs: [[0.3, 0.7]], value: [0] };
  const short = gaps({ logits: new Float32Array([0]), probs: new Float32Array([0.3]), value: new Float32Array([0]) }, ref, 0, line, 1);
  assert.equal(passes(short, line), false);
  const empty = await witness({ kind: "cpu", run: async () => ({ logits: new Float32Array(), probs: new Float32Array(), value: new Float32Array() }), dispose() {} },
    () => ({}) as Arrays, { ...ref, positions: [] }, line, 8, () => 0);
  assert.equal(empty.pass, false);
});
