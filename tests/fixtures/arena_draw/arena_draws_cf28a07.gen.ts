// The generator of arena_draws_cf28a07.json: run under node (type stripping) from a hexarena checkout at cf28a07,
// <hexarena@cf28a07> replaced by its path; provenance only, the tests read the JSON.
import { drawOpening, openingRegion } from '<hexarena@cf28a07>/packages/rules/src/opening.ts';
import { createRng } from '<hexarena@cf28a07>/packages/rules/test/helpers/prng.ts';
const out: Record<string, unknown> = { region: openingRegion.map((c) => [c.x, c.y]) };
for (const [seed, plies, n] of [[20261006, 5, 288], [0x0bee, 7, 400], [1, 9, 200], [7, 3, 50], [12345, 1, 3]]) {
  const rng = createRng(seed);
  out[`${seed}_${plies}`] = Array.from({ length: n }, () => drawOpening(plies, (b) => rng.int(b)).stones.map((s) => [s.x, s.y, s.player]));
}
const r = createRng(0x0bee);
out.ints = Array.from({ length: 10 }, () => r.int(18));
console.log(JSON.stringify(out));
