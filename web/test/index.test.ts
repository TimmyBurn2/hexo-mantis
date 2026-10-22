// The typed API's Worker handling, against a scripted fake Worker: a trap rejects the trapping call and respawns, calls
// wait out a respawn, respawns are capped, a failed first init cleans up, and dispose settles every call.
import assert from "node:assert/strict";
import { beforeEach, test } from "node:test";
import { createEngine, MAX_RESPAWNS } from "../src/index.ts";

type Msg = { id: number; op: string; args: unknown[] };
const workers: FakeWorker[] = [];
let script: (w: FakeWorker, msg: Msg) => { ok: true; value: unknown } | { ok: false; error: { name: string; message: string } } | null;

class FakeWorker {
  listeners: Array<(e: { data: unknown }) => void> = [];
  terminated = false;
  received: Msg[] = [];
  constructor(_url: string) {
    workers.push(this);
  }
  addEventListener(type: string, l: (e: { data: unknown }) => void) {
    if (type === "message") this.listeners.push(l);
  }
  postMessage(msg: Msg) {
    this.received.push(msg);
    const reply = script(this, msg);
    if (reply !== null) setTimeout(() => this.listeners.forEach((l) => l({ data: { id: msg.id, ...reply } })), 0);
  }
  terminate() {
    this.terminated = true;
  }
}
Object.assign(globalThis, { Worker: FakeWorker, location: { href: "http://127.0.0.1/" } });

const INFO = { backend: "cpu", leafMs: 1 };
const trap = { ok: false as const, error: { name: "RuntimeError", message: "unreachable" } };
beforeEach(() => {
  workers.length = 0;
});

test("a trapping call rejects with EngineCrashed, the engine respawns and counts it, and the next call reaches the new Worker", async () => {
  script = (w, m) => (m.op === "init" ? { ok: true, value: INFO } : w === workers[0] ? trap : { ok: true, value: { info: INFO } });
  const engine = await createEngine({ assets: "/pkg" });
  await assert.rejects(engine.playTurn({ sims: 8 }), { name: "EngineCrashed" });
  await engine.play([]);
  assert.equal(workers.length, 2);
  assert.ok(workers[0]!.terminated);
  assert.deepEqual(workers[1]!.received.map((m) => m.op), ["init", "play"]);
  assert.equal(engine.info().engineTraps, 1);
});

test("past the respawn cap the engine stays down and every call rejects without a new Worker", async () => {
  script = (_w, m) => (m.op === "init" ? { ok: true, value: INFO } : trap);
  const engine = await createEngine({ assets: "/pkg" });
  for (let i = 0; i <= MAX_RESPAWNS; i++) await assert.rejects(engine.playTurn({ sims: 8 }), { name: "EngineCrashed" });
  const spawned = workers.length;
  await assert.rejects(engine.playTurn({ sims: 8 }), { name: "EngineCrashed" });
  assert.equal(workers.length, spawned);
  assert.equal(spawned, MAX_RESPAWNS + 1);
});

test("a first init that fails rejects createEngine and terminates its Worker", async () => {
  script = () => ({ ok: false, error: { name: "NoBackendPassed", message: "[]" } });
  await assert.rejects(createEngine({ assets: "/pkg" }), { name: "NoBackendPassed" });
  assert.ok(workers[0]!.terminated);
});

test("dispose rejects the call in flight and every later call with AbortError", async () => {
  script = (_w, m) => (m.op === "init" ? { ok: true, value: INFO } : null);
  const engine = await createEngine({ assets: "/pkg" });
  const inFlight = engine.playTurn({ sims: 8 });
  engine.dispose();
  await assert.rejects(inFlight, { name: "AbortError" });
  await assert.rejects(engine.play([]), { name: "AbortError" });
});
