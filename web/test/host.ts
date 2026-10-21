// The node host the tests drive the Worker's handler through: files from the tiny stamp's export on disk, no GPU.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as Glue from "../pkg/mantis_player.js";
import type { Host } from "../src/worker.ts";

export const WEB = join(dirname(fileURLToPath(import.meta.url)), "..");
export const REPO = join(WEB, "..");
const fixture = process.env.MANTIS_WEB_FIXTURE;
if (fixture === undefined) throw new Error("MANTIS_WEB_FIXTURE names the tiny export (gate 21 writes it: tests/tools/_web_stamps.py)");
/** The tiny stamp's export: `net/index.json` beside its files. */
export const ASSETS = fixture;

const wasm = readFileSync(join(WEB, "pkg", "mantis_player_bg.wasm"));
Glue.initSync({ module: wasm });
export { Glue };

/** The node host, its fetches optionally rewritten (a manifest refusal's test edits the bytes it serves). */
export function nodeHost(edit: (path: string, bytes: Uint8Array<ArrayBuffer>) => Uint8Array<ArrayBuffer> = (_, b) => b): Host {
  let clock = 0;
  return {
    glue: async () => ({ glue: Glue, moduleBytes: wasm.byteLength }),
    fetchBytes: async (url) => {
      const bytes = new Uint8Array(readFileSync(url));
      return edit(url, bytes);
    },
    ort: async () => null,
    adapter: async () => null,
    now: () => (clock += 1),
    macrotask: () => new Promise((resolve) => setImmediate(resolve)),
  };
}

/** The golden positions as served requests, as the exporter reads them (`S` lines). */
export function goldens(n: number): Array<[Array<[number, number, number]>, number, number]> {
  const out: Array<[Array<[number, number, number]>, number, number]> = [];
  for (const line of readFileSync(join(REPO, "tests", "fixtures", "graph_build", "positions_r8.txt"), "utf8").split("\n")) {
    const [head, ...fields] = line.split(/\s+/);
    if (head !== "S") continue;
    const v = fields.map(Number);
    const stones = Array.from({ length: v[2]! }, (_, i) => [v[3 + 3 * i]!, v[4 + 3 * i]!, v[5 + 3 * i]!] as [number, number, number]);
    out.push([stones, v[0]!, v[1]!]);
    if (out.length === n) break;
  }
  return out;
}
