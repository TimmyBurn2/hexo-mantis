// The engine's typed API on the main thread: every call is a message to the Worker that owns the engine (contract #13).
import type { WitnessRow } from "./backends.ts";

export type Cell = { q: number; r: number };
export type Position = { stones: Array<Cell & { p: 1 | -1 }>; toMove: 1 | -1; stonesLeft: 1 | 2 };
export type Budget = { turnMs: number } | { sims: number } | { unlimited: true };
export type Clock = { turnMs: number } | { unlimited: true };
export type Child = { cell: [number, number]; visits: number; prior: number | null; q: number | null };
export type Stone = {
  cell: [number, number]; sims: number; stopped: boolean; decided: boolean; inherited: number; rootVisits: number;
  rootValue: number | null; top: Child[]; tactics: Record<string, number> | null;
};
export type TurnResult = {
  stones: Stone[]; simsSet: number; floorOverClock: boolean; budgetPoolCapped: boolean; wallMs: number;
  servedLeaves: number; rows: Record<string, number>;
};
export type EngineInfo = {
  backend: "webgpu" | "cpu"; adapter: string | null; witness: WitnessRow; rows: WitnessRow[]; leafMs: number;
  rootMs: number; minTurnMs: number | null; moduleBytes: number; peakBytes: number; stamp: Record<string, unknown>;
  engineTraps: number;
};
export type Request = { id: number; op: string; args: unknown[] };
export type Reply = { id: number; ok: true; value: unknown } | { id: number; ok: false; error: { name: string; message: string } };

/** A refusal from the engine or the Worker, named as contract #13 names it. */
export class EngineError extends Error {
  constructor(name: string, message: string) {
    super(message);
    this.name = name;
  }
}

export interface Engine {
  info(): EngineInfo;
  newGame(opening: Position, clock: Clock): Promise<void>;
  play(stones: Cell[]): Promise<void>;
  playTurn(budget: Budget): Promise<TurnResult>;
  analyse(position: Position, budget: Budget): Promise<TurnResult>;
  abort(): void;
  dispose(): void;
}

/** An engine in its own Worker, its files fetched from `assets` alone; a trap rebuilds it and counts it. */
/** Respawns after traps in a row past which the engine is left down: a Worker that cannot start stops retrying. */
export const MAX_RESPAWNS = 3;

export async function createEngine(opts: { assets: string; backend?: "auto" | "webgpu" | "cpu" }): Promise<Engine> {
  // Absolute, so the Worker (whose own URL is the base inside it) fetches from the same place as the page.
  const assets = new URL(opts.assets, location.href).href.replace(/\/$/, "");
  let worker: Worker;
  let info: EngineInfo;
  let traps = 0;
  let inARow = 0;
  let next = 1;
  let down: string | null = null;
  let ready = false;
  const pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();

  const settleAll = (name: string, message: string) => {
    for (const waiting of pending.values()) waiting.reject(new EngineError(name, message));
    pending.clear();
  };
  const spawn = async (): Promise<void> => {
    worker = new Worker(`${assets}/src/worker.js`, { type: "module" });
    worker.addEventListener("message", (event: MessageEvent<Reply>) => {
      const reply = event.data;
      const waiting = pending.get(reply.id);
      if (waiting === undefined) return;
      pending.delete(reply.id);
      if (reply.ok) {
        waiting.resolve(reply.value);
      } else if (reply.error.name === "RuntimeError") {
        waiting.reject(new EngineError("EngineCrashed", reply.error.message));
        void crashed(reply.error.message);
      } else {
        waiting.reject(new EngineError(reply.error.name, reply.error.message));
      }
    });
    worker.addEventListener("error", (event) => void crashed(event.message));
    info = (await call("init", assets, opts.backend ?? "auto")) as EngineInfo;
  };
  const call = (op: string, ...args: unknown[]): Promise<unknown> => new Promise((resolve, reject) => {
    if (down !== null) return reject(new EngineError("EngineCrashed", down));
    const id = next++;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, op, args } satisfies Request);
  });
  const crashed = async (message: string): Promise<void> => {
    settleAll("EngineCrashed", message);
    worker.terminate();
    traps += 1;
    inARow += 1;
    if (!ready || inARow > MAX_RESPAWNS) {
      down = ready ? `the engine trapped ${inARow} times in a row: ${message}` : message;
      return;
    }
    await spawn().catch((e: Error) => void (down = `${e.name}: ${e.message}`));
  };
  const withInfo = async <T>(op: string, ...args: unknown[]): Promise<T> => {
    const value = (await call(op, ...args)) as T & { info?: EngineInfo };
    if (value?.info) info = value.info;
    inARow = 0;
    return value;
  };

  try {
    await spawn();
    ready = true;
  } catch (e) {
    worker!.terminate();
    throw e;
  }
  return {
    info: () => ({ ...info, engineTraps: traps }),
    newGame: (opening, clock) => withInfo("newGame", opening, clock),
    play: (stones) => withInfo("play", stones),
    playTurn: (budget) => withInfo("playTurn", budget),
    analyse: (position, budget) => withInfo("analyse", position, budget),
    abort: () => worker.postMessage({ id: 0, op: "abort", args: [] } satisfies Request),
    dispose: () => {
      settleAll("AbortError", "the engine was disposed");
      worker.terminate();
    },
  };
}
