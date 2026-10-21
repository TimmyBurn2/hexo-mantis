// The HeXO-Arena analysis slot's adapter: htttx basic_websocket v1-alpha packets over postMessage, as the arena's
// `apps/web/src/analysis/worker-source.ts` and `packages/contract/src/htttx.ts` speak them at develop 8817871a.
import type { Budget, Cell, Engine, Position, TurnResult } from "./index.ts";

export type Side = "x" | "o";
export type WireCell = { q: number; r: number; p: Side };
export type Port = { postMessage(message: unknown): void; addEventListener(type: "message", listener: (e: MessageEvent) => void): void;
  start?: () => void };

/** The capabilities the slot requires before it reads any position. */
export const capabilities = (version: string) => ({
  meta: { version },
  basic_websocket: { versions: { "v1-alpha": { free_setup: true, resettable_state: true, dual_sided: true, request_id: true,
    interruptible: true } } },
});
/** The slot's largest request: every deepening step fits a pool sized for it, so a position never regrows it. */
export const ANALYSIS_CLOCK_MS = 5_000;
// The fixed order a winning turn's unplaced second piece is drawn from: the first empty neighbour of the win.
const NEIGHBOURS: Array<[number, number]> = [[1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1]];

const sign = (side: Side) => (side === "x" ? 1 : -1);
const key = (q: number, r: number) => `${q},${r}`;

/** The move response for a turn: two pieces (a winning first stone padded by its first empty neighbour) and the
 * root's value, x-positive as htttx counts it (a root solver's decided stone is a proven win for the mover). */
export function response(turn: TurnResult, side: Side, occupied: Set<string>, requestId: number | undefined): unknown {
  const pieces = turn.stones.map((s) => ({ q: s.cell[0], r: s.cell[1] }));
  if (pieces.length === 1) {
    const [q, r] = turn.stones[0]!.cell;
    const taken = new Set([...occupied, key(q, r)]);
    const free = NEIGHBOURS.map(([dq, dr]) => ({ q: q + dq, r: r + dr })).find((c) => !taken.has(key(c.q, c.r)))!;
    pieces.push(free);
  }
  const last = [...turn.stones].reverse().find((s) => s.rootValue !== null);
  const won = turn.stones.length === 1 || turn.stones.some((s) => s.decided);
  const value = won ? 1 : Math.max(-1, Math.min(1, last?.rootValue ?? 0));
  return { type: "move_response", move: { pieces, evaluation: { heuristic: sign(side) * value } },
    ...(requestId === undefined ? {} : { request_id: requestId }) };
}

/** Serve the slot on `port` with `engine`: capabilities first, then each setup and move request in order. */
export function attachArena(port: Port, engine: Engine, version: string): void {
  let cells: WireCell[] = [];
  let fresh = false;
  let outstanding: number | undefined;
  const interrupted = new Set<number>();
  const occupied = new Set<string>();
  let queue: Promise<void> = Promise.resolve();

  const request = async (packet: { side: Side; previous: Array<{ side: Side; pieces: Cell[] }>; move_time_limit?: number;
    request_id?: number }): Promise<void> => {
    // A request interrupted while it waited is never started: the slot would count its answer a stray.
    if (packet.request_id !== undefined && interrupted.delete(packet.request_id)) return;
    const unseen = packet.previous.flatMap((t) => t.pieces.map((c) => ({ ...c, p: t.side })))
      .filter((c) => !occupied.has(key(c.q, c.r)));
    if (fresh) {
      const stones = [...cells, ...unseen].map((c) => ({ q: c.q, r: c.r, p: sign(c.p) as 1 | -1 }));
      const position: Position = { stones, toMove: sign(packet.side) as 1 | -1, stonesLeft: stones.length === 0 ? 1 : 2 };
      await engine.newGame(position, { turnMs: ANALYSIS_CLOCK_MS });
      occupied.clear();
      fresh = false;
    } else if (unseen.length > 0) {
      await engine.play(unseen);
    }
    for (const c of [...cells, ...unseen]) occupied.add(key(c.q, c.r));
    cells = [];
    const budget: Budget = packet.move_time_limit === undefined ? { unlimited: true } : { turnMs: packet.move_time_limit * 1000 };
    outstanding = packet.request_id;
    try {
      const turn = await engine.playTurn(budget);
      const answer = response(turn, packet.side, occupied, packet.request_id);
      for (const s of turn.stones) occupied.add(key(s.cell[0], s.cell[1]));
      port.postMessage(answer);
    } catch (e) {
      // An interrupted request is answered with nothing, as the slot expects; any other refusal times the slot out.
      if (!(e instanceof Error && e.name === "AbortError")) console.error("mantis: move request refused", e);
    } finally {
      outstanding = undefined;
    }
  };

  port.addEventListener("message", (event) => {
    const packet = event.data as { type?: string; board?: { cells: WireCell[] }; request_id?: number };
    if (packet.type === "setup") {
      queue = queue.then(() => {
        cells = packet.board?.cells ?? [];
        fresh = true;
      });
    } else if (packet.type === "interrupt") {
      if (packet.request_id === undefined || packet.request_id === outstanding) engine.abort();
      else interrupted.add(packet.request_id);
    } else if (packet.type === "move_request") {
      queue = queue.then(() => request(packet as Parameters<typeof request>[0]));
    }
  });
  port.start?.();
  port.postMessage(capabilities(version));
}
