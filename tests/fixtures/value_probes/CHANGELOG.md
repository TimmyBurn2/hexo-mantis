# value_probes fixtures — CHANGELOG

Frozen goldens gating the WP9 model oracles (dist65 codec, 234-probe value-health
metric freeze, small-net forward drift). Re-bump discipline: a fixture is
re-frozen ONLY with a new version row here naming the cause + the anchor/source it
was regenerated from; the fixtures-manifest test (`tests/test_fixtures_manifest.py`)
FAILS — never skips — if any committed fixture drifts from its manifest sha.

| ver | date | cause | anchor / source |
|---|---|---|---|
| v1 | 2026-07-18 | initial freeze (WP9 model port) | old-side capture `wp/WP9/oldside/` (torch 2.11.0, MKL/AVX512, CPU/1-thread/fp32/deterministic; OLD commit d769f46) |

## Contents

- `dist65_golden.json` — 23 `(z → two_hot fp32)` pairs + `(support_bin → decoded)` +
  `(z → decode(log(two_hot)))` pairs. Gates O1 (dist65 primitives). COPIED from
  old-side capture #1.
- `decoded_v.npz` + `metrics.json` — decoded-v arrays on the 234 loss + 651 safe
  positions and the frozen M1–M4 (M1=-0.20448957, M2=0.06981187, M3=0.87121719,
  M4=0.02457757). Gates O2a (metric-math golden). COPIED from old-side capture #2.
- `probe_set_v1.jsonl` (234 loss rows, sha `7899fa13…`) / `negatives_v1.jsonl`
  (651 safe rows, sha `8faa6af7…`) — the frozen probe/negative sets. Gate O2/O2c.
  COPIED from old-side capture #2.
- `forward/small_{cnn_scalar,cnn_dist65,cnn_aux_chain,gnn}.pt` — small-net forward
  drift guards (O_bench A-committed). NEW-GENERATED new-side (a self-contained
  regression reference for the ported code, NOT the old-side parity reference,
  which is Tier-1 and lives in `wp/WP9/oldside/`). Fixed seed + fixed input +
  saved weights → the test rebuilds, loads, forwards, and asserts a match.

## Metric definitions (M1–M4, frozen)

- M1 `mean_v_on_losses` — mean decoded-v over the LOSS positions.
- M2 `ece` — ECE, 10 equal-width bins on P_win=(v+1)/2 over loss∪safe.
- M3 `decoded_auc` (scalar arm) = AUC(-v, loss=1 vs safe=0); `tail_mass_auc` (dist
  arm) = AUC(tail-mass); one is null per arm.
- M4 `false_pessimism` — fraction of SAFE controls with decoded-v ≤ -0.5.
