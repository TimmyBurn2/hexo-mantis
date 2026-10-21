// The Worker that owns the wasm engine and its backend: the message handler, the manifest's checks, selection, and
// the one loop both backends turn through (contract #13). Exported as a handler, so node drives it without a Worker.
import type * as Ort from "onnxruntime-web";
import type * as Glue from "../pkg/mantis_player.js";
import {
  arraysOf, cpuBackend, ortBackend, select, witness,
  type Arrays, type Backend, type DeviceWitness, type Manifest, type WitnessRow,
} from "./backends.ts";
import { EngineError, type Budget, type Cell, type Clock, type EngineInfo, type Position, type Reply, type Request,
  type TurnResult } from "./index.ts";

export const MANIFEST_FORMAT = "mantis-web-manifest-v2";
// A WebGPU call's edges at most, so its [E, H] message and index tensors stay inside a GPU's budget.
export const EDGE_CAP = 120_000;

/** What the handler reaches the outside world through; the browser's are below, a test passes its own. */
export type Host = {
  glue(assets: string): Promise<{ glue: typeof Glue; moduleBytes: number }>;
  fetchBytes(url: string): Promise<Uint8Array<ArrayBuffer>>;
  ort(assets: string): Promise<typeof Ort | null>;
  adapter(): Promise<string | null>;
  now(): number;
  macrotask(): Promise<void>;
};

/** A 1-instruction simd128 module: an engine built with +simd128 cannot instantiate where this does not validate. */
const SIMD_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0,
  65, 0, 253, 15, 253, 98, 11]);

/** The stones of `position` as moves in turn cadence (x one, then two each), each side's in the order given; the
 * engine's rules check the result, so a position no cadence reaches is refused there. */
export function cadence(position: Position): Cell[] {
  const own = (p: 1 | -1) => position.stones.filter((s) => s.p === p).map(({ q, r }) => ({ q, r }));
  const [x, o] = [own(1), own(-1)];
  const moves: Cell[] = [];
  let side: 1 | -1 = 1;
  let left = 1;
  while (x.length + o.length > 0) {
    const next = (side === 1 ? x : o).shift();
    if (next === undefined) throw new EngineError("IllegalStone", "the position's stones fit no turn cadence");
    moves.push(next);
    left -= 1;
    if (left === 0) [side, left] = [side === 1 ? -1 : 1, 2];
  }
  if (side !== position.toMove || left !== position.stonesLeft) {
    throw new EngineError("IllegalStone", `the cadence leaves ${side} to move with ${left}, not ${position.toMove} with ${position.stonesLeft}`);
  }
  return moves;
}

const flat = (cells: Cell[]): Int32Array => Int32Array.from(cells.flatMap((c) => [c.q, c.r]));
const camel = (v: unknown): unknown => Array.isArray(v) ? v.map(camel) : v !== null && typeof v === "object"
  ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k.replace(/_(.)/g, (_, c: string) => c.toUpperCase()), camel(x)]))
  : v;

async function sha256(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

/** A manifest this engine carries, its fp32 files fetched and their sha256 checked. */
async function checked(host: Host, glue: typeof Glue, assets: string): Promise<{ manifest: Manifest; file: (suffix: string) => Promise<Uint8Array<ArrayBuffer>> }> {
  const index = JSON.parse(new TextDecoder().decode(await host.fetchBytes(`${assets}/net/index.json`))) as { manifest: string };
  const manifest = JSON.parse(new TextDecoder().decode(await host.fetchBytes(`${assets}/net/${index.manifest}`))) as Manifest;
  const refuse = (why: string) => new EngineError("Refused", `manifest: ${why}`);
  if (manifest.format !== MANIFEST_FORMAT) throw refuse(`format ${manifest.format}, this engine reads ${MANIFEST_FORMAT}`);
  if (manifest.encoding.registry_sha !== glue.registry_sha()) throw refuse("exported under another encoding registry");
  if (manifest.deploy.resolved.search_kind !== "puct") throw refuse(`deploy kind ${manifest.deploy.resolved.search_kind}`);
  const own = JSON.parse(glue.player_constants()) as Record<string, unknown>;
  for (const [k, v] of Object.entries(own)) if (manifest.deploy.player[k] !== v) throw refuse(`deploy.player.${k} is not this engine's ${String(v)}`);
  const file = async (suffix: string): Promise<Uint8Array<ArrayBuffer>> => {
    const row = manifest.files.find((f) => f.name.endsWith(suffix));
    if (row === undefined) throw refuse(`no ${suffix} file`);
    const bytes = await host.fetchBytes(`${assets}/net/${row.name}`);
    if ((await sha256(bytes)) !== row.sha256) throw refuse(`${row.name} does not match its sha256`);
    return bytes;
  };
  return { manifest, file };
}

/** The handler for `{id, op, args}` messages; `abort` answers nothing and is heard between batches. */
export function handler(host: Host): (msg: Request) => Promise<Reply | null> {
  let glue: typeof Glue;
  let engine: Glue.Engine;
  let backend: Backend;
  let manifest: Manifest;
  let info: EngineInfo;
  let feat = 0;
  let moduleBytes = 0;
  let busy = false;
  let aborted = false;

  const named = (name: string, message: string) => new EngineError(name, message);

  async function init(assets: string, choice: "auto" | "webgpu" | "cpu"): Promise<EngineInfo> {
    if (!WebAssembly.validate(SIMD_PROBE)) throw named("Unsupported", "this browser's WebAssembly has no SIMD");
    ({ glue, moduleBytes } = await host.glue(assets));
    const net = await checked(host, glue, assets);
    manifest = net.manifest;
    feat = manifest.encoding.node_feat_dim;
    const d = manifest.deploy.resolved;
    const t = d.tactics;
    // The CPU path arms the 16-sim floor under a short clock, so the engine is built for the backend that won.
    const build_engine = (cpuPath: boolean) => new glue.Engine(manifest.encoding.name, cpuPath, d.c_puct, d.fpu_reduction,
      d.quiescence_enabled, d.quiescence_blend_2, d.leaf_batch_size,
      t ? Uint32Array.from([t.leaf_turns, t.leaf_nodes, t.root_turns, t.root_nodes]) : undefined,
      t?.audit ? Uint32Array.from([t.audit.turns, t.audit.nodes, t.audit.k, t.audit.m, t.audit.total_nodes]) : undefined,
      t?.audit?.mode);
    const ref = JSON.parse(new TextDecoder().decode(await net.file(".witness.json"))) as DeviceWitness;
    const build = (positions: DeviceWitness["positions"]): Arrays => {
      const stones = Int32Array.from(positions.flatMap(([s]) => s.flat()));
      const batch = glue.web_batch(manifest.encoding.name, stones, Uint32Array.from(positions.map(([s]) => s.length)),
        Int32Array.from(positions.map(([, side]) => side)), Int32Array.from(positions.map(([, , left]) => left)));
      const arrays = arraysOf(batch, feat);
      batch.free();
      return arrays;
    };
    const adapter = choice === "cpu" ? null : await host.adapter();
    const gpu = async (): Promise<Backend | null> => {
      const ort = adapter === null ? null : await host.ort(assets);
      if (ort === null) return null;
      const onnx = await net.file(".fp32.onnx");
      const session = await ort.InferenceSession.create(onnx, { executionProviders: ["webgpu"], graphOptimizationLevel: "all" });
      return ortBackend(ort, session, EDGE_CAP);
    };
    const cpu = async (): Promise<Backend> => {
      const bytes = await net.file(".weights.bin");
      const table = manifest.files.find((f) => f.name.endsWith(".weights.bin"))!.tensors!;
      engine = build_engine(true);
      engine.load_weights(table.map((r) => r.name).join("\n"), Uint32Array.from(table.flatMap((r) => [r.rows, r.cols, r.offset])),
        new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4), feat);
      return cpuBackend(engine, glue);
    };
    const candidates = choice === "cpu" ? [cpu] : choice === "webgpu" ? [gpu] : [gpu, cpu];
    const line = manifest.witness.lines.fp32!;
    const chosen = await select(candidates, (b) => witness(b, build, ref, line, d.leaf_batch_size, host.now));
    if (!("backend" in chosen)) throw named("NoBackendPassed", JSON.stringify(chosen.rows));
    backend = chosen.backend;
    if (backend.kind === "webgpu") engine = build_engine(false);
    // The witness's cost a leaf until a turn measures its own; root phases fold into it, as after a turn.
    engine.set_costs(Math.max(chosen.row.ms / chosen.row.leaves, 1e-3), 0);
    info = describe(adapter, chosen.row, chosen.rows);
    return info;
  }

  function describe(adapter: string | null, row: WitnessRow, rows: WitnessRow[]): EngineInfo {
    return {
      backend: backend.kind, adapter: backend.kind === "webgpu" ? adapter : null, witness: row, rows,
      leafMs: row.ms / row.leaves, rootMs: 0, minTurnMs: engine.min_turn_ms ?? null, moduleBytes,
      peakBytes: glue.memory_bytes(), stamp: manifest.stamp, engineTraps: 0,
    };
  }

  function newGame(opening: Position, clock: Clock): void {
    const turnMs = "turnMs" in clock ? clock.turnMs : undefined;
    engine.new_game(flat(cadence(opening)), undefined, turnMs);
  }

  async function playTurn(budget: Budget): Promise<TurnResult> {
    if (busy) throw named("OutOfStep", "a turn is in play");
    busy = true;
    aborted = false;
    try {
      engine.begin_turn(host.now(), "sims" in budget ? budget.sims : undefined, "turnMs" in budget ? budget.turnMs : undefined);
      while (engine.next_batch(host.now())) {
        if (backend.kind === "cpu") {
          engine.forward_pending();
          engine.submit_pending(host.now());
        } else {
          const pending = engine.pending_batch();
          const out = await backend.run(arraysOf(pending, feat));
          pending.free();
          engine.submit(out.probs, out.value, host.now());
        }
        await host.macrotask();
        if (aborted) {
          engine.abort();
          throw named("AbortError", "the turn was aborted; the board is as before it");
        }
      }
      const turn = camel(JSON.parse(engine.turn_json()!)) as Omit<TurnResult, "rows">;
      if (turn.servedLeaves > 0) info = { ...info, leafMs: turn.wallMs / turn.servedLeaves };
      info = { ...info, minTurnMs: engine.min_turn_ms ?? null, peakBytes: glue.memory_bytes() };
      return { ...turn, rows: JSON.parse(engine.rows_json()) as Record<string, number> };
    } finally {
      busy = false;
    }
  }

  const ops: Record<string, (...args: never[]) => unknown> = {
    init: (assets: string, choice: "auto" | "webgpu" | "cpu") => init(assets, choice),
    info: () => info,
    newGame: (opening: Position, clock: Clock) => (newGame(opening, clock), { info }),
    play: (stones: Cell[]) => (engine.play(flat(stones)), { info }),
    playTurn: async (budget: Budget) => ({ ...(await playTurn(budget)), info }),
    analyse: async (position: Position, budget: Budget) => {
      if (busy) throw named("OutOfStep", "a turn is in play");
      newGame(position, "turnMs" in budget ? { turnMs: budget.turnMs } : { unlimited: true });
      return { ...(await playTurn(budget)), info };
    },
    dispose: () => {
      backend?.dispose();
      engine?.free();
      return null;
    },
  };

  return async (msg) => {
    if (msg.op === "abort") {
      aborted = true;
      return null;
    }
    const op = ops[msg.op];
    try {
      if (op === undefined) throw named("OutOfStep", `no operation ${msg.op}`);
      return { id: msg.id, ok: true, value: await (op as (...a: unknown[]) => unknown)(...msg.args) };
    } catch (e) {
      const error = e instanceof Error ? { name: e.name, message: e.message } : { name: "Error", message: String(e) };
      return { id: msg.id, ok: false, error };
    }
  };
}

/** The browser's host: files from the integrator's origin only, ORT's runtime from `assets/ort/`. */
export const browserHost: Host = {
  glue: async (assets) => {
    const glue = (await import(`${assets}/pkg/mantis_player.js`)) as typeof Glue & { default: (o: { module_or_path: BufferSource }) => Promise<unknown> };
    const bytes = await browserHost.fetchBytes(`${assets}/pkg/mantis_player_bg.wasm`);
    await glue.default({ module_or_path: bytes });
    return { glue, moduleBytes: bytes.byteLength };
  },
  fetchBytes: async (url) => {
    const res = await fetch(url);
    if (!res.ok) throw new EngineError("Refused", `${url}: HTTP ${res.status}`);
    return new Uint8Array(await res.arrayBuffer());
  },
  ort: async (assets) => {
    const ort = (await import(`${assets}/ort/ort.webgpu.bundle.min.mjs`)) as typeof Ort;
    ort.env.wasm.wasmPaths = `${assets}/ort/`;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    return ort;
  },
  adapter: async () => {
    if (!("gpu" in navigator)) return null;
    // A fresh GPU process can answer its first request empty; a few short retries before the CPU path.
    let adapter: GPUAdapter | null = null;
    for (let attempt = 0; attempt < 5 && adapter === null; attempt++) {
      adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (adapter === null) await new Promise((resolve) => setTimeout(resolve, 200));
    }
    // A software adapter (SwiftShader) is no GPU: the CPU path's own forward beats it.
    if (adapter === null || (adapter.info as GPUAdapterInfo & { isFallbackAdapter?: boolean }).isFallbackAdapter) return null;
    return `${adapter.info.vendor} ${adapter.info.architecture}`.trim();
  },
  now: () => performance.now(),
  macrotask: () => new Promise((resolve) => {
    const channel = new MessageChannel();
    channel.port1.onmessage = () => {
      channel.port1.close();
      resolve();
    };
    channel.port2.postMessage(0);
  }),
};

declare const WorkerGlobalScope: unknown;
if (typeof WorkerGlobalScope !== "undefined") {
  const handle = handler(browserHost);
  const scope = globalThis as unknown as { onmessage: (e: MessageEvent<Request>) => void; postMessage(r: Reply): void };
  scope.onmessage = (event) => void handle(event.data).then((reply) => reply && scope.postMessage(reply));
}
