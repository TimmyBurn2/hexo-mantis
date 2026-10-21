// The WebGPU backend's per-call edge cap: a batch split at graph boundaries gives the outputs of the whole batch, on a
// stub session whose outputs depend on each graph alone (so a re-basing error shows as a wrong value).
import assert from "node:assert/strict";
import { test } from "node:test";
import * as ort from "onnxruntime-web";
import { ortBackend, splitBatch, type Arrays, type Session } from "../src/backends.ts";
import { Glue, goldens } from "./host.ts";

function batch(n: number): Arrays {
  const positions = goldens(n);
  const b = Glue.web_batch("gnn_axis_r8", Int32Array.from(positions.flatMap(([s]) => s.flat())),
    Uint32Array.from(positions.map(([s]) => s.length)), Int32Array.from(positions.map(([, side]) => side)),
    Int32Array.from(positions.map(([, , left]) => left)));
  const ints = ["src", "dst", "code", "legal", "node_graph", "legal_graph", "divisor", "real", "mean_mask", "mean_den"] as const;
  const a = { x: b.x(), feat: 11, ...Object.fromEntries(ints.map((k) => [k, b.i32(k)])) } as Arrays;
  b.free();
  return a;
}

/** A graph's value is its edges' code sum over its nodes' first features; a legal row's logit its node's in-degree. */
const stub = (calls: number[]): Session => ({
  async run(feeds) {
    const f = Object.fromEntries(Object.entries(feeds).map(([k, t]) => [k, (t as ort.Tensor).data])) as Record<string, any>;
    calls.push(f.src.length);
    const graphs = f.mean_den.length;
    const value = new Float32Array(graphs);
    const degree = new Float32Array(f.node_graph.length);
    for (let e = 0; e < f.src.length; e++) {
      value[f.node_graph[f.dst[e]]]! += f.code[e] * f.x[f.src[e] * 11];
      degree[f.dst[e]]! += 1;
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
