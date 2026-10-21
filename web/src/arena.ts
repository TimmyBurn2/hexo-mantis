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

/** The first empty cell around (q, r): its neighbours in a fixed order, then each wider ring, all within the rules'
 * radius of a stone; a winning first stone's partner, which the arena checks for legality and never places. */
export function padding(q: number, r: number, taken: Set<string>): Cell {
  for (const [dq, dr] of NEIGHBOURS) if (!taken.has(key(q + dq, r + dr))) return { q: q + dq, r: r + dr };
  for (let ring = 2; ring <= 8; ring++) {
    for (let dq = -ring; dq <= ring; dq++) {
      for (let dr = -ring; dr <= ring; dr++) {
        if (Math.max(Math.abs(dq), Math.abs(dr), Math.abs(dq + dr)) === ring && !taken.has(key(q + dq, r + dr))) {
          return { q: q + dq, r: r + dr };
        }
      }
    }
  }
  throw new Error("no empty cell within the rules' radius");
}

/** The move response for a turn that was due `due` stones: two pieces (a winning first stone padded) and the root's
 * value, x-positive as htttx counts it (a decided stone is a proven win, as is a turn that ended early). */
export function response(turn: TurnResult, side: Side, occupied: Set<string>, requestId: number | undefined, due: number): unknown {
  const pieces = turn.stones.map((s) => ({ q: s.cell[0], r: s.cell[1] }));
  if (pieces.length === 1) {
    const [q, r] = turn.stones[0]!.cell;
    pieces.push(padding(q, r, new Set([...occupied, key(q, r)])));
  }
  const last = [...turn.stones].reverse().find((s) => s.rootValue !== null);
  const won = turn.stones.length < due || turn.stones.some((s) => s.decided);
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
    const id = packet.request_id;
    // An interrupted request is never started, played on or answered: the slot would count its answer a stray.
    const dropped = () => id !== undefined && interrupted.has(id);
    outstanding = id;
    try {
      if (dropped()) return;
      const unseen = packet.previous.flatMap((t) => t.pieces.map((c) => ({ ...c, p: t.side })))
        .filter((c) => !occupied.has(key(c.q, c.r)));
      let due = 2;
      if (fresh) {
        const stones = [...cells, ...unseen].map((c) => ({ q: c.q, r: c.r, p: sign(c.p) as 1 | -1 }));
        due = stones.length === 0 ? 1 : 2;
        fresh = false;
        await engine.newGame({ stones, toMove: sign(packet.side) as 1 | -1, stonesLeft: due as 1 | 2 }, { turnMs: ANALYSIS_CLOCK_MS });
      } else if (unseen.length > 0) {
        await engine.play(unseen);
      }
      for (const c of [...cells, ...unseen]) occupied.add(key(c.q, c.r));
      cells = [];
      if (dropped()) return;
      const budget: Budget = packet.move_time_limit === undefined ? { unlimited: true } : { turnMs: packet.move_time_limit * 1000 };
      const turn = await engine.playTurn(budget);
      const answer = response(turn, packet.side, occupied, id, due);
      for (const s of turn.stones) occupied.add(key(s.cell[0], s.cell[1]));
      if (!dropped()) port.postMessage(answer);
    } catch (e) {
      // A refused request is answered with nothing and the slot times it out; the queue goes on to the next one.
      if (!(e instanceof Error && e.name === "AbortError")) console.error("mantis: move request refused", e);
    } finally {
      outstanding = undefined;
      if (id !== undefined) interrupted.delete(id);
    }
  };

  port.addEventListener("message", (event) => {
    const packet = event.data as { type?: string; board?: { cells: WireCell[] }; request_id?: number };
    if (packet.type === "setup") {
      queue = queue.then(() => {
        cells = packet.board?.cells ?? [];
        fresh = true;
        occupied.clear();
      });
    } else if (packet.type === "interrupt") {
      if (packet.request_id !== undefined) interrupted.add(packet.request_id);
      if (packet.request_id === undefined || packet.request_id === outstanding) engine.abort();
    } else if (packet.type === "move_request") {
      queue = queue.then(() => request(packet as Parameters<typeof request>[0]));
    }
  });
  port.start?.();
  port.postMessage(capabilities(version));
}
