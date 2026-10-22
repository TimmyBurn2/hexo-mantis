// W2 in wasm: the engine's RustForward against the fp32 ONNX on ORT's one-thread CPU kernels, on the goldens at B 1
// and 8 and the export's openings, held to the fp32 line; fed every divisor as 1 it must miss (planted break).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";
import * as ort from "onnxruntime-web";
import { buildBatch, cpuBackend, gaps, INT_INPUTS, passes, type Arrays, type DeviceWitness, type Manifest, type NetOut } from "../src/backends.ts";
import { ASSETS, Glue, goldens } from "./host.ts";

const index = JSON.parse(readFileSync(join(ASSETS, "net", "index.json"), "utf8")) as { manifest: string };
const manifest = JSON.parse(readFileSync(join(ASSETS, "net", index.manifest), "utf8")) as Manifest;
const file = (suffix: string) => join(ASSETS, "net", manifest.files.find((f) => f.name.endsWith(suffix))!.name);
const feat = manifest.encoding.node_feat_dim;
const line = manifest.witness.lines.fp32!;
const d = manifest.deploy.resolved;

ort.env.wasm.numThreads = 1;

function engine(): Glue.Engine {
  const e = new Glue.Engine(manifest.encoding.name, true, d.c_puct, d.fpu_reduction, d.quiescence_enabled, d.quiescence_blend_2,
    d.leaf_batch_size, undefined, undefined, undefined);
  const table = manifest.files.find((f) => f.name.endsWith(".weights.bin"))!.tensors!;
  const bytes = readFileSync(file(".weights.bin"));
  e.load_weights(table.map((r) => r.name).join("\n"), Uint32Array.from(table.flatMap((r) => [r.rows, r.cols, r.offset])),
    new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4), feat);
  return e;
}

const build = (positions: DeviceWitness["positions"]): Arrays => buildBatch(Glue, manifest.encoding.name, positions, feat);

const rust = (e: Glue.Engine, a: Arrays): Promise<NetOut> => cpuBackend(e, Glue).run(a);

/** The ONNX's outputs as a witness reference, one row per graph. */
function reference(out: NetOut, a: Arrays, positions: DeviceWitness["positions"]): DeviceWitness {
  const split = (v: Float32Array) => Array.from({ length: a.mean_den.length }, (_, g) =>
    Array.from(v.filter((_, i) => a.legal_graph[i] === g)));
  return { positions, logits: split(out.logits), probs: split(out.probs), value: Array.from(out.value) };
}

async function rows(e: Glue.Engine, batchSize: number, drop: boolean) {
  const session = await ort.InferenceSession.create(file(".fp32.onnx"), { executionProviders: ["wasm"] });
  const positions = [...goldens(64), ...(JSON.parse(readFileSync(file(".witness.json"), "utf8")) as DeviceWitness).positions.slice(0, 5)];
  const out = [];
  for (let from = 0; from < positions.length; from += batchSize) {
    const slice = positions.slice(from, from + batchSize);
    const a = build(slice);
    const feeds: Record<string, ort.Tensor> = { x: new ort.Tensor("float32", a.x, [a.x.length / feat, feat]) };
    for (const k of INT_INPUTS) feeds[k] = new ort.Tensor("int32", a[k], [a[k].length]);
    const o = await session.run(feeds);
    const ref = reference({ logits: o.logits!.data as Float32Array, probs: o.probs!.data as Float32Array,
      value: o.value!.data as Float32Array }, a, slice);
    const fed = drop ? { ...a, divisor: a.divisor.map(() => 1) } : a;
    out.push(gaps(await rust(e, fed), ref, 0, line, slice.length));
  }
  await session.release();
  return out;
}

test("W2: RustForward in wasm holds the fp32 line against the ONNX on one CPU thread at B 1 and 8", async () => {
  const e = engine();
  for (const b of [1, 8]) {
    for (const row of await rows(e, b, false)) assert.ok(passes(row, line), `B ${b}: ${JSON.stringify(row)}`);
  }
  e.free();
});

test("W2's planted break: the forward fed every divisor as 1 misses the line", async () => {
  const e = engine();
  const missed = (await rows(e, 8, true)).filter((row) => !passes(row, line));
  assert.ok(missed.length > 0, "dropping the degree divisor went unseen");
  e.free();
});
