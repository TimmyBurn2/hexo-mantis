// The Worker's handler on the tiny stamp's export: the manifest's refusals, the witness that admits the CPU backend,
// turns through the one loop, an abort heard between batches, and the budget in situ.
import assert from "node:assert/strict";
import { test } from "node:test";
import type { Reply } from "../src/index.ts";
import { cadence, handler, type Host } from "../src/worker.ts";
import { ASSETS, nodeHost } from "./host.ts";

type Ok = Extract<Reply, { ok: true }>;
const call = async (handle: ReturnType<typeof handler>, op: string, ...args: unknown[]) => (await handle({ id: 1, op, args }))!;
const ok = (reply: Reply) => {
  assert.ok(reply.ok, JSON.stringify(reply));
  return (reply as Ok).value as Record<string, unknown>;
};
const started = async (host: Host = nodeHost()) => {
  const handle = handler(host);
  return { handle, info: ok(await call(handle, "init", ASSETS, "cpu")) };
};
const EMPTY = { stones: [], toMove: 1, stonesLeft: 1 } as const;

test("the CPU backend passes its witness on the export's 16 positions and is taken", async () => {
  const { info } = await started();
  const row = info.witness as Record<string, number | boolean | string>;
  assert.equal(info.backend, "cpu");
  assert.equal(row.pass, true);
  assert.equal(row.leaves, 16);
  assert.ok((row.argmax_compared as number) > 0);
  assert.ok((info.leafMs as number) > 0 && (info.peakBytes as number) > 0 && (info.moduleBytes as number) > 0);
});

test("a manifest of another format, another engine's constants, or a file off its sha256 is refused by name", async () => {
  const edits: Array<(m: Record<string, any>) => void> = [
    (m) => (m.format = "mantis-web-manifest-v1"),
    (m) => (m.deploy.player.cache_entries = 4096),
    (m) => (m.encoding.registry_sha = "0".repeat(64)),
    (m) => (m.files.find((f: { name: string }) => f.name.endsWith(".weights.bin")).sha256 = "0".repeat(64)),
  ];
  for (const edit of edits) {
    const host = nodeHost((path, bytes) => {
      if (!path.endsWith(".manifest.json")) return bytes;
      const m = JSON.parse(new TextDecoder().decode(bytes));
      edit(m);
      return new TextEncoder().encode(JSON.stringify(m)) as Uint8Array<ArrayBuffer>;
    });
    const reply = await call(handler(host), "init", ASSETS, "cpu");
    assert.ok(!reply.ok && reply.error.name === "Refused", JSON.stringify(reply));
  }
});

test("a game plays turn by turn: the origin, then two stones, each placed by the engine's rules", async () => {
  const { handle } = await started();
  ok(await call(handle, "newGame", EMPTY, { turnMs: 10_000 }));
  const first = ok(await call(handle, "playTurn", { sims: 32 }));
  assert.deepEqual((first.stones as Array<{ cell: number[] }>).map((s) => s.cell), [[0, 0]]);
  ok(await call(handle, "play", [{ q: 1, r: 0 }, { q: 2, r: -1 }]));
  const second = ok(await call(handle, "playTurn", { sims: 32 }));
  const stones = second.stones as Array<{ cell: number[]; sims: number; stopped: boolean }>;
  assert.equal(stones.length, 2);
  for (const s of stones) assert.ok(s.sims === 32 || s.stopped || s.sims === 0, JSON.stringify(s));
  assert.ok(((second.rows as Record<string, number>).cache_calls ?? 0) > 0);
  const refused = await call(handle, "play", [{ q: 0, r: 0 }]);
  assert.ok(!refused.ok && refused.error.name === "IllegalStone");
});

test("an abort is heard between batches: the turn rejects with AbortError, and the board is as before it", async () => {
  let handle!: ReturnType<typeof handler>;
  let batches = 0;
  const host = nodeHost();
  const macrotask = host.macrotask;
  host.macrotask = async () => {
    batches += 1;
    if (batches === 2) await handle({ id: 0, op: "abort", args: [] });
    await macrotask();
  };
  ({ handle } = await started(host));
  ok(await call(handle, "newGame", { stones: [{ q: 0, r: 0, p: 1 }], toMove: -1, stonesLeft: 2 }, { unlimited: true }));
  const aborted = await call(handle, "playTurn", { sims: 64 });
  assert.ok(!aborted.ok && aborted.error.name === "AbortError", JSON.stringify(aborted));
  const after = ok(await call(handle, "playTurn", { sims: 32 }));
  assert.equal((after.stones as unknown[]).length, 2, "the side to move is unchanged, so it still plays two stones");
});

test("a clock budget sets its sims from the device's measured cost, at or above the CPU floor", async () => {
  const { handle, info } = await started();
  ok(await call(handle, "newGame", { stones: [{ q: 0, r: 0, p: 1 }], toMove: -1, stonesLeft: 2 }, { turnMs: 2_000 }));
  const turn = ok(await call(handle, "playTurn", { turnMs: 2_000 }));
  assert.ok((turn.simsSet as number) >= 16, JSON.stringify(turn));
  assert.equal(typeof info.minTurnMs, "number");
});

test("a second turn while one is in play is OutOfStep", async () => {
  const { handle } = await started();
  ok(await call(handle, "newGame", { stones: [{ q: 0, r: 0, p: 1 }], toMove: -1, stonesLeft: 2 }, { unlimited: true }));
  const [a, b] = await Promise.all([call(handle, "playTurn", { sims: 64 }), call(handle, "playTurn", { sims: 64 })]);
  assert.ok(a.ok);
  assert.ok(!b.ok && b.error.name === "OutOfStep");
});

test("a position becomes moves in turn cadence, and one no cadence reaches is refused", () => {
  const x = (q: number, r: number) => ({ q, r, p: 1 as const });
  const o = (q: number, r: number) => ({ q, r, p: -1 as const });
  assert.deepEqual(cadence({ stones: [o(1, 0), x(0, 0), o(2, 0), x(5, 0)], toMove: 1, stonesLeft: 1 }), [
    { q: 0, r: 0 }, { q: 1, r: 0 }, { q: 2, r: 0 }, { q: 5, r: 0 }]);
  assert.throws(() => cadence({ stones: [x(0, 0), x(1, 0)], toMove: -1, stonesLeft: 2 }), { name: "IllegalStone" });
});

test("in auto, a WebGPU candidate that throws is a failed row and the CPU path is taken", async () => {
  const host = nodeHost();
  host.adapter = async () => ({ handle: {}, label: "stub" });
  host.ort = async () => { throw new Error("no WebGPU runtime here"); };
  const info = ok(await call(handler(host), "init", ASSETS, "auto"));
  const rows = info.rows as Array<{ backend: string; pass: boolean; error?: string }>;
  assert.equal(info.backend, "cpu");
  assert.deepEqual(rows.map((r) => [r.backend, r.pass]), [["webgpu", false], ["cpu", true]]);
  assert.match(rows[0]!.error ?? "", /no WebGPU runtime/);
});

test("a turn that fails mid-loop for any reason leaves the board as before it and the engine usable", async () => {
  const host = nodeHost();
  const macrotask = host.macrotask;
  let calls = 0;
  host.macrotask = async () => {
    if (++calls === 2) throw new Error("the device was lost");
    await macrotask();
  };
  const { handle } = await started(host);
  ok(await call(handle, "newGame", { stones: [{ q: 0, r: 0, p: 1 }], toMove: -1, stonesLeft: 2 }, { unlimited: true }));
  const failed = await call(handle, "playTurn", { sims: 64 });
  assert.ok(!failed.ok && /device was lost/.test(failed.error.message), JSON.stringify(failed));
  const after = ok(await call(handle, "playTurn", { sims: 32 }));
  assert.equal((after.stones as unknown[]).length, 2);
});
