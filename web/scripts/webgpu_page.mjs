// In-page: the shipped ONNX on ORT-web's WebGPU EP against the served reference (W1 at B 1 and 8, and the openings),
// then ms a leaf for WebGPU and the engine's RustForward on the same batches, then the p99 leaf repeated at B 8.
// Query: manifest=<file in /net/>, timing=<positions timed a B>, passes=<timed passes>.
import { arraysOf, gaps, INT_INPUTS, passes as holds } from "/dist/src/backends.js";

const q = new URLSearchParams(location.search);
const result = { crossOriginIsolated, userAgent: navigator.userAgent };

async function main() {
  const glue = await import("/dist/pkg/mantis_player.js");
  await glue.default({ module_or_path: "/dist/pkg/mantis_player_bg.wasm" });
  const ort = await import("/dist/ort/ort.webgpu.bundle.min.mjs");
  ort.env.wasm.wasmPaths = "/dist/ort/";
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.logLevel = "verbose";
  result.ortVersion = ort.env.versions?.web ?? ort.env.versions?.common;
  // A fresh browser's first request can come back empty while its GPU process starts; ask a few times.
  let adapter = null;
  for (let attempt = 0; attempt < 10 && adapter === null; attempt++) {
    const a = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (a && !a.info.isFallbackAdapter) adapter = a;
    else await new Promise((resolve) => setTimeout(resolve, 500));
  }
  if (!adapter) throw new Error("no hardware WebGPU adapter");
  result.adapter = { vendor: adapter.info.vendor, architecture: adapter.info.architecture, description: adapter.info.description };
  ort.env.webgpu.adapter = adapter;

  const manifest = await (await fetch(`/net/${q.get("manifest")}`)).json();
  const file = (suffix) => manifest.files.find((f) => f.name.endsWith(suffix));
  const feat = manifest.encoding.node_feat_dim;
  const line = manifest.witness.lines.fp32;
  const t0 = performance.now();
  const session = await ort.InferenceSession.create(`/net/${file(".fp32.onnx").name}`,
    { executionProviders: ["webgpu"], graphOptimizationLevel: "all", logSeverityLevel: 0, logVerbosityLevel: 0 });
  result.createMs = performance.now() - t0;
  ort.env.logLevel = "warning";

  const d = manifest.deploy.resolved;
  const engine = new glue.Engine(manifest.encoding.name, true, d.c_puct, d.fpu_reduction, d.quiescence_enabled,
    d.quiescence_blend_2, d.leaf_batch_size, undefined, undefined, undefined);
  const table = file(".weights.bin").tensors;
  const weights = new Float32Array(await (await fetch(`/net/${file(".weights.bin").name}`)).arrayBuffer());
  engine.load_weights(table.map((r) => r.name).join("\n"), Uint32Array.from(table.flatMap((r) => [r.rows, r.cols, r.offset])),
    weights, feat);

  const ref = await (await fetch("/ref.json")).json();
  const build = (positions) => {
    const batch = glue.web_batch(manifest.encoding.name, Int32Array.from(positions.flatMap(([s]) => s.flat())),
      Uint32Array.from(positions.map(([s]) => s.length)), Int32Array.from(positions.map(([, side]) => side)),
      Int32Array.from(positions.map(([, , left]) => left)));
    const arrays = arraysOf(batch, feat);
    batch.free();
    return arrays;
  };
  const feeds = (a) => {
    const f = { x: new ort.Tensor("float32", a.x, [a.x.length / feat, feat]) };
    for (const k of INT_INPUTS) f[k] = new ort.Tensor("int32", a[k], [a[k].length]);
    return f;
  };
  const webgpu = async (a) => {
    const o = await session.run(feeds(a));
    return { logits: o.logits.data, probs: o.probs.data, value: o.value.data };
  };
  const rust = (a) => {
    const b = new glue.Batch(a.x, a.src, a.dst, a.code, a.legal, a.node_graph, a.legal_graph, a.divisor, a.real, a.mean_mask, a.mean_den);
    const o = engine.forward_batch(b);
    b.free();
    return { logits: o.logits(), probs: o.probs(), value: o.value() };
  };
  const chunks = (positions, b) => Array.from({ length: Math.ceil(positions.length / b) }, (_, i) => positions.slice(i * b, (i + 1) * b));

  // W1: every golden at B 1 and 8, and the openings as one batch, against the served reference at that B.
  result.witness = {};
  for (const [label, positions, b, reference] of [["b1", ref.goldens, 1, ref.ref["1"]], ["b8", ref.goldens, 8, ref.ref["8"]],
    ["openings", ref.openings, ref.openings.length, ref.ref.openings]]) {
    const rows = [];
    let from = 0;
    for (const chunk of chunks(positions, b)) {
      const out = await webgpu(build(chunk));
      rows.push(gaps(out, { ...reference, positions }, from, line));
      from += chunk.length;
    }
    const row = { logits: Math.max(...rows.map((r) => r.logits)), probs: Math.max(...rows.map((r) => r.probs)),
      value: Math.max(...rows.map((r) => r.value)), argmax_agreed: rows.reduce((n, r) => n + r.argmax_agreed, 0),
      argmax_compared: rows.reduce((n, r) => n + r.argmax_compared, 0) };
    result.witness[label] = { ...row, pass: holds(row, line) };
  }

  // Timing: the same positions for both backends, spread over stone counts; one warm pass, then `passes` timed ones.
  const every = Math.max(1, Math.floor(ref.goldens.length / Number(q.get("timing") ?? 64)));
  const timed = ref.goldens.filter((_, i) => i % every === 0);
  const passes = Number(q.get("passes") ?? 5);
  result.timing = {};
  for (const b of [1, 8]) {
    const batches = chunks(timed, b).map(build);
    for (const [name, run] of [["webgpu", webgpu], ["rust", async (a) => rust(a)]]) {
      for (const a of batches) await run(a);
      const ms = [];
      for (let p = 0; p < passes; p++) {
        const s = performance.now();
        for (const a of batches) await run(a);
        ms.push(performance.now() - s);
      }
      ms.sort((x, y) => x - y);
      result.timing[`${name}_b${b}`] = { leaves: timed.length, msPerLeafMedian: ms[Math.floor(ms.length / 2)] / timed.length,
        msPerLeafMin: ms[0] / timed.length, msPerLeafMax: ms[ms.length - 1] / timed.length };
    }
  }

  // The goldens' p99 leaf, eight to a batch, repeated while the runner samples GPU memory.
  const sizes = ref.goldens.map((p, i) => [build([p]).x.length / feat, i]).sort((x, y) => x[0] - y[0]);
  const [nodes, at] = sizes[Math.min(sizes.length - 1, Math.floor(sizes.length * 0.99))];
  const p99 = build(Array(8).fill(ref.goldens[at]));
  result.p99 = { golden: at, nodes, edges: p99.src.length, startedAt: Date.now() };
  for (let i = 0; i < 20; i++) await webgpu(p99);
  result.p99.endedAt = Date.now();
  await session.release();
  engine.free();
}

main().then(() => { result.ok = true; }, (err) => { result.ok = false; result.error = String(err?.stack ?? err); })
  .finally(() => { window.__result = result; });
