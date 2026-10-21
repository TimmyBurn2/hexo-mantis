// In-page: the shipped WebGPU path (the Worker's host for the adapter and ORT, the backend with its edge cap, the
// witness's rows) against the served reference — W1 at B 1 and 8 and the openings — then ms a leaf for it and for the
// engine's RustForward on the same batches, then the goldens' p99 leaf eight to a batch, repeated.
// Query: manifest=<file in /net/>, timing=<positions timed a B>, passes=<timed passes>.
import { arraysOf, cpuBackend, ortBackend, witness } from "/dist/src/backends.js";
import { browserHost, EDGE_CAP } from "/dist/src/worker.js";

const q = new URLSearchParams(location.search);
const result = { crossOriginIsolated, userAgent: navigator.userAgent };

async function main() {
  const glue = await import("/dist/pkg/mantis_player.js");
  await glue.default({ module_or_path: "/dist/pkg/mantis_player_bg.wasm" });
  const adapter = await browserHost.adapter();
  if (adapter === null) throw new Error("no hardware WebGPU adapter");
  result.adapter = adapter.label;
  const ort = await browserHost.ort("/dist");
  ort.env.webgpu.adapter = adapter.handle;
  ort.env.logLevel = "verbose";
  result.ortVersion = ort.env.versions?.web ?? ort.env.versions?.common;

  const manifest = await (await fetch(`/net/${q.get("manifest")}`)).json();
  const file = (suffix) => manifest.files.find((f) => f.name.endsWith(suffix));
  const feat = manifest.encoding.node_feat_dim;
  const line = manifest.witness.lines.fp32;
  const t0 = performance.now();
  const session = await ort.InferenceSession.create(`/net/${file(".fp32.onnx").name}`,
    { executionProviders: ["webgpu"], graphOptimizationLevel: "all", logSeverityLevel: 0, logVerbosityLevel: 0 });
  result.createMs = performance.now() - t0;
  ort.env.logLevel = "warning";
  const webgpu = ortBackend(ort, session, EDGE_CAP);

  const d = manifest.deploy.resolved;
  const engine = new glue.Engine(manifest.encoding.name, true, d.c_puct, d.fpu_reduction, d.quiescence_enabled,
    d.quiescence_blend_2, d.leaf_batch_size, undefined, undefined, undefined);
  const table = file(".weights.bin").tensors;
  const weights = new Float32Array(await (await fetch(`/net/${file(".weights.bin").name}`)).arrayBuffer());
  engine.load_weights(table.map((r) => r.name).join("\n"), Uint32Array.from(table.flatMap((r) => [r.rows, r.cols, r.offset])),
    weights, feat);
  const rust = cpuBackend(engine, glue);

  const ref = await (await fetch("/ref.json")).json();
  const build = (positions) => {
    const batch = glue.web_batch(manifest.encoding.name, Int32Array.from(positions.flatMap(([s]) => s.flat())),
      Uint32Array.from(positions.map(([s]) => s.length)), Int32Array.from(positions.map(([, side]) => side)),
      Int32Array.from(positions.map(([, , left]) => left)));
    const arrays = arraysOf(batch, feat);
    batch.free();
    return arrays;
  };
  const now = () => performance.now();

  result.witness = {
    b1: await witness(webgpu, build, { positions: ref.goldens, ...ref.ref["1"] }, line, 1, now),
    b8: await witness(webgpu, build, { positions: ref.goldens, ...ref.ref["8"] }, line, 8, now),
    openings: await witness(webgpu, build, { positions: ref.openings, ...ref.ref.openings }, line, ref.openings.length, now),
  };

  // Timing: the same positions for both backends, spread over stone counts; one warm pass, then `passes` timed ones.
  const every = Math.max(1, Math.floor(ref.goldens.length / Number(q.get("timing") ?? 64)));
  const timed = ref.goldens.filter((_, i) => i % every === 0);
  const passes = Number(q.get("passes") ?? 5);
  result.timing = {};
  for (const b of [1, 8]) {
    const batches = Array.from({ length: Math.ceil(timed.length / b) }, (_, i) => build(timed.slice(i * b, (i + 1) * b)));
    for (const backend of [webgpu, rust]) {
      for (const a of batches) await backend.run(a);
      const ms = [];
      for (let p = 0; p < passes; p++) {
        const s = performance.now();
        for (const a of batches) await backend.run(a);
        ms.push(performance.now() - s);
      }
      ms.sort((x, y) => x - y);
      result.timing[`${backend.kind}_b${b}`] = { leaves: timed.length, msPerLeafMedian: ms[Math.floor(ms.length / 2)] / timed.length,
        msPerLeafMin: ms[0] / timed.length, msPerLeafMax: ms[ms.length - 1] / timed.length };
    }
  }

  // The goldens' p99 leaf, eight to a batch through the shipped backend (so split at its edge cap), repeated while the
  // runner samples GPU memory.
  const sizes = ref.goldens.map((p, i) => [build([p]).x.length / feat, i]).sort((x, y) => x[0] - y[0]);
  const [nodes, at] = sizes[Math.min(sizes.length - 1, Math.floor(sizes.length * 0.99))];
  const p99 = build(Array(8).fill(ref.goldens[at]));
  result.p99 = { golden: at, nodes, edges: p99.src.length, edgeCap: EDGE_CAP, startedAt: Date.now() };
  for (let i = 0; i < 20; i++) await webgpu.run(p99);
  result.p99.endedAt = Date.now();
  webgpu.dispose();
  engine.free();
}

main().then(() => { result.ok = true; }, (err) => { result.ok = false; result.error = String(err?.stack ?? err); })
  .finally(() => { window.__result = result; });
