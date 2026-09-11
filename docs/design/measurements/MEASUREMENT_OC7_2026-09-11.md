# MEASUREMENT — CARD-OC7-OVERRUN discriminated (R348(b)), 2026-09-11

Instrument: the OC-7 row itself,
`tests/train/test_clean_stop_save.py::test_a_clean_run_at_the_minted_bound_leaves_one_stamped_checkpoint`,
under a 600 s wall-clock cap with the whole process group killed at the cap, a py-spy dump
(python + native) at 300 s, and the run's own JSONL as the step clock. Every number below is
read off those artefacts, not estimated.

# VERDICT: **HOST, not code.** The dev box emulates bf16 on the CPU trainer; the box does not.

---

## 1. The first fact was a wave-2 regression, not the overrun

At `b117e657` the row **fails in 46 s** before any timing question arises:
`pool_drain.py:106` unpacks a 10-tuple; DELETE-1 (`3dd20b49`) shrank the Rust `GameResultRow`
to 8 fields and updated the `.pyi`, but not the consumer, and every default-tier fake scripted
10. Only the real engine drain — the integration tier — could see it, and that tier had not
run since. Fixed at `92643671`; `tests/selfplay/test_drain_row_shape_parity.py` now binds the
producer's declaration, the stub, the consumer's unpack and every fake to one arity. Everything
below is on the fixed tree.

## 2. (b)(i) — hang or slow: SLOW

| host | `n_workers` | trainer steps observed | mean s/step | at the 300 s dump |
|---|---|---|---|---|
| dev box (Ryzen 7 3700X, 16 threads, AVX2 only) | 14 (minted) | 5 by 314 s | **~60** | main thread in `at::native::cpublas` bf16 GEMM inside `backward_accumulate`; 9 cores busy; not blocked |
| dev box | 1 (the 2026-08-01 premise) | 2 by 85 s | **~40** | — |
| dev box, tree at `9363522c` (the 2026-08-01 M-0 commit) | 1 | 4 by 161 s | **~40** | — |
| box `vast` (Ryzen 9 9900X, 24 threads, AVX-512 BF16) | 14 (minted) | 50, **PASS in 178.4 s** | ~2.9 incl. rounds | n/a |

The 110 s sampled profile on the dev box: 44% of all samples in the trainer backward, 45% in
the inference server's GINE forward, both CPU torch. Trainer step 1 carried a batch of 32 rows
= 603,628 edges / 23,470 nodes in one microbatch. The docstring's own three numbers imply
~1.6 s/step on 2026-08-01 (474.6−319.5 s over 100 steps).

## 3. (b)(ii)/(iii) — host versus code

The M-0 tree and HEAD run at the same ~40 s/step on the dev box today, and HEAD runs at
178 s on the box, so a `git bisect` over `mcts/` has nothing to find and was not run. The
mechanism was then measured directly rather than inferred:

```
threads=8  fp32 71.7 ms | bf16 5197.9 ms (72.5x) | autocast-bf16 5206.0 ms (72.6x)
```

— one `(600000, 128) @ (128, 128)` GEMM, the GINE edge-MLP shape at the observed edge count,
in the project venv on the dev box. LAW-06 pins bf16 autocast on the graph path and the
trainer applies it on `device_type="cpu"` too (`trainer/core.py`, `_autocast_enabled =
amp_dtype == bfloat16`); on a CPU without AVX-512 BF16 every such GEMM takes ATen's generic
path. `/proc/cpuinfo` on the dev box carries `avx2` and no `avx512*`; the box carries
`avx512_bf16`. Same torch wheel on both dates and both hosts (`2.11.0+cpu`, pinned since
2026-07-25; the box runs `+cu128` for the trainer only where the config says cuda — the smoke
config says cpu on both).

Contributing, not driving: the 2026-08-31 mint `1203f740` put `selfplay.n_workers: 14` into
the smoke config the row drives; the row's bound was measured at 1 worker. On the dev box
that is the difference between 40 and 60 s/step.

Not recorded anywhere: which physical machine took the 2026-08-01 numbers. The register says
"the dev box"; this dev box's CPU cannot have produced 1.6 s/step under bf16 autocast.

## 4. What this settles and what it leaves to the operator

- The docstring's rule stands and yields a bound on the box: 50 measures 178 s there, so no
  re-aim is needed where the tier can run at all.
- On an AVX2 host the rule yields NO member of {200, 100, 50, 32, 16} — 16 steps alone is
  ~640 s. The integration tier is therefore not runnable on this dev box under LAW-06's pin.
  Which of (a) the tier runs on the box, (b) the row is `slow`-marked, or (c) LAW-06 gains a
  CPU carve-out is a ruling, not a dispatcher's choice; (c) is a law amendment.
- (b)(iv): the tier's remaining 49 tests were run on the box with `-v --durations=0`; the
  reading is in `docs/governance/STATE.md` beside the card.

## 5. Artefacts

Dev box: scratchpad `oc7/{head_local,head_fixed_local}/` (meta, pytest log, py-spy dumps,
`profile_raw.txt`), `oc7/drive_w1.log`, `oc7/drive_m0.log`. Box: `/workspace/oc7/box_head/`
and `/workspace/oc7/tier_box.log`. Host identity lines for the box are in
`/workspace/box_setup_wave3.log` step 0.
