// The analysis-slot adapter on packets transcribed from HeXO-Arena develop 8817871a (`packages/contract/src/htttx.ts`,
// `apps/web/src/analysis/worker-source.ts`), against a recording engine: nothing of the arena's runs here.
import assert from "node:assert/strict";
import { test } from "node:test";
import { ANALYSIS_CLOCK_MS, attachArena, capabilities, type Port } from "../src/arena.ts";
import type { Budget, Engine, Position, TurnResult } from "../src/index.ts";

/** The packets worker-source posts for one deepening step of a position: a fresh setup, then a move request. */
const SETUP = { type: "setup", board: { cells: [{ q: 0, r: 0, p: "x" }, { q: 1, r: 0, p: "o" }, { q: 1, r: -1, p: "o" }] } };
const REQUEST = { type: "move_request", side: "x", previous: [], move_time_limit: 0.25, request_id: 1 };
const INTERRUPT = { type: "interrupt", request_id: 1 };

const stone = (q: number, r: number, extra: Partial<TurnResult["stones"][number]> = {}) => ({
  cell: [q, r] as [number, number], sims: 32, stopped: false, decided: false, inherited: 0, rootVisits: 32,
  rootValue: 0.25, top: [], tactics: null, ...extra,
});
const turnOf = (stones: TurnResult["stones"]): TurnResult => ({ stones, simsSet: 32, floorOverClock: false,
  budgetPoolCapped: false, wallMs: 1, servedLeaves: 4, rows: {} });

function harness(next: () => Promise<TurnResult>) {
  const listeners: Array<(e: MessageEvent) => void> = [];
  const posted: unknown[] = [];
  const calls: Array<[string, unknown?, unknown?]> = [];
  const port: Port = { postMessage: (m) => posted.push(m), addEventListener: (_, l) => listeners.push(l) };
  const engine: Engine = {
    info: () => { throw new Error("unused"); },
    newGame: async (p: Position, c) => void calls.push(["newGame", p, c]),
    play: async (s) => void calls.push(["play", s]),
    playTurn: async (b: Budget) => (calls.push(["playTurn", b]), next()),
    analyse: async () => { throw new Error("unused"); },
    abort: () => void calls.push(["abort"]),
    dispose: () => {},
  };
  attachArena(port, engine, "test");
  const send = (data: unknown) => listeners.forEach((l) => l({ data } as MessageEvent));
  const settle = () => new Promise((r) => setTimeout(r, 0));
  return { posted, calls, send, settle };
}

test("the engine declares the five capabilities the slot requires before it reads a position", () => {
  const { posted } = harness(async () => turnOf([]));
  assert.deepEqual(posted, [capabilities("test")]);
  const declared = (posted[0] as ReturnType<typeof capabilities>).basic_websocket.versions["v1-alpha"];
  assert.deepEqual(Object.keys(declared).sort(), ["dual_sided", "free_setup", "interruptible", "request_id", "resettable_state"]);
});

test("a setup and a move request start a game at the setup, sized for the slot, and answer two pieces and a heuristic", async () => {
  const { posted, calls, send, settle } = harness(async () => turnOf([stone(2, -1), stone(-1, 1, { rootValue: -0.5 })]));
  send(SETUP);
  send(REQUEST);
  await settle();
  const [, position, clock] = calls[0]!;
  assert.deepEqual(position, { stones: [{ q: 0, r: 0, p: 1 }, { q: 1, r: 0, p: -1 }, { q: 1, r: -1, p: -1 }], toMove: 1, stonesLeft: 2 });
  assert.deepEqual(clock, { turnMs: ANALYSIS_CLOCK_MS });
  assert.deepEqual(calls[1], ["playTurn", { turnMs: 250 }]);
  assert.deepEqual(posted[1], { type: "move_response", move: { pieces: [{ q: 2, r: -1 }, { q: -1, r: 1 }],
    evaluation: { heuristic: -0.5 } }, request_id: 1 });
});

test("o's heuristic is signed x-positive, and an absent time limit is an unlimited turn", async () => {
  const { posted, calls, send, settle } = harness(async () => turnOf([stone(2, -1), stone(-1, 1, { rootValue: 0.75 })]));
  send({ type: "setup", board: { cells: [{ q: 0, r: 0, p: "x" }] } });
  send({ type: "move_request", side: "o", previous: [], request_id: 7 });
  await settle();
  assert.deepEqual(calls[1], ["playTurn", { unlimited: true }]);
  assert.deepEqual((posted[1] as { move: { evaluation: unknown } }).move.evaluation, { heuristic: -0.75 });
});

test("a turn whose first stone wins answers that stone and its first empty neighbour, valued a win", async () => {
  const { posted, send, settle } = harness(async () => turnOf([stone(2, -1, { decided: true, rootValue: null })]));
  send({ type: "setup", board: { cells: [{ q: 0, r: 0, p: "x" }, { q: 3, r: -1, p: "o" }] } });
  send({ ...REQUEST, side: "x" });
  await settle();
  assert.deepEqual((posted[1] as { move: unknown }).move, { pieces: [{ q: 2, r: -1 }, { q: 2, r: 0 }], evaluation: { heuristic: 1 } });
});

test("an interrupt aborts the request in flight and nothing is answered for it", async () => {
  let release!: (t: TurnResult) => void;
  const { posted, calls, send, settle } = harness(() => new Promise<TurnResult>((_, reject) => {
    release = () => reject(Object.assign(new Error("aborted"), { name: "AbortError" }));
  }));
  send(SETUP);
  send(REQUEST);
  await settle();
  send(INTERRUPT);
  assert.deepEqual(calls.at(-1), ["abort"]);
  release(turnOf([]));
  await settle();
  assert.equal(posted.length, 1, "only the capabilities were posted");
});

test("a request interrupted before it started is never played", async () => {
  let release!: () => void;
  let turns = 0;
  const { calls, send, settle } = harness(() => new Promise<TurnResult>((resolve) => {
    turns += 1;
    release = () => resolve(turnOf([stone(2, -1), stone(-1, 1)]));
  }));
  send(SETUP);
  send(REQUEST);
  send(SETUP);
  send({ ...REQUEST, request_id: 2 });
  await settle();
  send({ type: "interrupt", request_id: 2 });
  release();
  await settle();
  assert.equal(turns, 1, `request 2 was played after its interrupt: ${JSON.stringify(calls)}`);
});
