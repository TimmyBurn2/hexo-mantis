// The demo: a person plays x against the engine on a local static server, beside the per-turn LAW-18 rows. It reads
// no run and connects to nothing but the package's own files.
import { createEngine, type Cell, type Engine, type TurnResult } from "../src/index.ts";

const SIZE = 14;
const DIRS: Array<[number, number]> = [[1, 0], [0, 1], [1, -1]];
const $ = (id: string) => document.getElementById(id)!;
const board = new Map<string, 1 | -1>();
const key = (q: number, r: number) => `${q},${r}`;
let engine: Engine;
let mine: Cell[] = [];
let busy = true;
let over = false;

const wins = (q: number, r: number, p: 1 | -1) => DIRS.some(([dq, dr]) => {
  let run = 1;
  for (const s of [1, -1]) for (let i = 1; board.get(key(q + s * i * dq, r + s * i * dr)) === p; i++) run++;
  return run >= 6;
});
const left = () => (board.size === 0 ? 1 : 2) - mine.length;

function rows(table: string, entries: Array<[string, unknown]>): void {
  $(table).replaceChildren(...entries.map(([k, v]) => {
    const tr = document.createElement("tr");
    for (const text of [k, typeof v === "number" ? String(+v.toFixed(3)) : String(v)]) tr.insertCell().textContent = text;
    return tr;
  }));
}

function draw(): void {
  const cells = new Set<string>();
  const reach = Math.max(6, ...[...board.keys()].map((k) => Math.max(...k.split(",").map((n) => Math.abs(+n))) + 3));
  for (let q = -reach; q <= reach; q++) for (let r = -reach; r <= reach; r++) if (Math.abs(q + r) <= reach) cells.add(key(q, r));
  const scale = Math.min(1, 9 / reach);
  $("board").innerHTML = [...cells].map((k) => {
    const [q, r] = k.split(",").map(Number) as [number, number];
    const [x, y] = [SIZE * scale * Math.sqrt(3) * (q + r / 2), -SIZE * scale * 1.5 * r];
    const p = board.get(k);
    return `<circle cx="${x}" cy="${y}" r="${SIZE * scale * 0.8}" class="${p === 1 ? "x" : p === -1 ? "o" : "cell"}" data-q="${q}" data-r="${r}"/>`;
  }).join("");
}

function report(turn: TurnResult, turnMs: number): void {
  const info = engine.info();
  rows("turn", [
    ["sims set", turn.simsSet], ["sims spent", turn.stones.map((s) => s.sims).join(" / ")],
    ["stops fired", turn.stones.filter((s) => s.stopped).length], ["reuse inherited", turn.stones.map((s) => s.inherited).join(" / ")],
    ["cache hits / served", `${turn.rows.cache_hits} / ${turn.rows.cache_served}`],
    ["tactics decided", turn.stones.filter((s) => s.decided).length], ["wall ms", turn.wallMs],
    ["turn over 0.8T", turn.wallMs > 0.8 * turnMs], ["floor over clock", turn.floorOverClock],
    ["pool capped", turn.budgetPoolCapped], ["leaf ms", info.leafMs], ["root ms", info.rootMs], ["min turn ms", info.minTurnMs ?? "—"],
  ]);
  rows("engine", [["backend", info.backend], ["adapter", info.adapter ?? "—"], ["witness", info.witness.pass ? "pass" : "MISS"],
    ["witness |Δ logit|", info.witness.logits], ["module bytes", info.moduleBytes], ["peak memory bytes", info.peakBytes],
    ["engine traps", info.engineTraps]]);
}

async function reply(): Promise<void> {
  busy = true;
  const turnMs = Number(($("clock") as HTMLInputElement).value);
  $("status").textContent = "mantis is thinking…";
  await engine.play(mine);
  mine = [];
  const turn = await engine.playTurn({ turnMs });
  for (const s of turn.stones) {
    board.set(key(...s.cell), -1);
    if (wins(...s.cell, -1)) over = true;
  }
  draw();
  report(turn, turnMs);
  $("status").textContent = over ? "mantis wins." : "your turn: place two stones.";
  busy = false;
}

async function restart(): Promise<void> {
  board.clear();
  mine = [];
  over = false;
  await engine.newGame({ stones: [], toMove: 1, stonesLeft: 1 }, { turnMs: Number(($("clock") as HTMLInputElement).value) });
  draw();
  $("status").textContent = "your turn: x opens with one stone.";
  busy = false;
}

$("board").addEventListener("click", (event) => {
  const target = event.target as SVGElement;
  if (busy || over || target.dataset.q === undefined) return;
  const [q, r] = [Number(target.dataset.q), Number(target.dataset.r)];
  if (board.has(key(q, r))) return;
  board.set(key(q, r), 1);
  mine.push({ q, r });
  draw();
  if (wins(q, r, 1)) {
    over = true;
    $("status").textContent = "you win.";
  } else if (left() === 0) {
    void reply().catch((e: Error) => ($("status").textContent = `${e.name}: ${e.message}`));
  }
});
$("restart").addEventListener("click", () => void restart());

engine = await createEngine({ assets: new URL("..", import.meta.url).href });
await restart();
