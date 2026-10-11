# DENSE-1 — a hex-conv model kind behind the seam: researched, then designed (2026-10-11)

Status: **DESIGN** (the DENSE-1-DESIGN packet under R391(e); CARD-DENSE-1). No code, config or mint change rides
this document, and nothing in it orders code or a run. The rival trees were read statically and our tree read-only.

It decides the frame, the planes, the trunk and heads, the seam and the bootstrap of a dense hex-conv kind. It prices
them and lays the build in legs with gates. It builds only on R391(d)'s trigger (run12 misses its line at 84k) and
the architect's ruling.

- **Evidence:** `mantis-records/research/DENSE1_2026-10-11.md` and `mantis-records/research/dense1_2026-10-11/`. The
  evidence file holds dated sources (S1–S7). The directory holds the three research reports, the arithmetic, and a
  fresh red team's review with this revision's dispositions (`review/`).
- **Tree:** code read at d0a7346c. The doc sits on RUN12-GO's first commit 65a04419, which is docs-only over it.

## 0. Decisions — one screen

**Bad news first: three of the packet's premises are wrong (§1).**
- **Six's nets are not 10×96.** Only gen 10 is. Gen 30, our X, is 10×128; gen 455, rung 16's opponent, is 20×128 at
  7.4 GFLOP per position.
- **Bubble's buckets are canvas sides of 24–256, not 24–128 cells.** Its crop is the box of every stone and every
  legal cell.
- **"≈ 1.7 GFLOP" has no source.** It matches our 4×128's training-path edge term (1.75 GF). Served, the GNN costs
  0.23 GF (4×128) and 0.73 GF (4×256, run12's net).

Two more facts shape the design. The 65-bin value head is ours; Six reads two logits and Bubble one. Neither rival
ever ran an A/B on its size, planes, line features or symmetry handling, so their choices are choices, not results.

1. **Frame — the fence canvas.**
   - The net sees exactly the GNN's set: every stone and every radius-8 legal cell. That set is masked inside its
     smallest axial box, with sides in multiples of 8. All convolutions, norms and pools are masked to it.
   - So no centre, window or crop edge exists. The encoding commutes with all 12 D6 elements, and every legal cell
     gets a prior.
   - **Far mode** (a box past `S_max`, sized by L0 so the corpus never needs it): every stone stays in with a halo of
     4, and the cells beyond share one `far` logit (Bubble's rule).
   - **The fence plane plus masking re-adopts the canvas-realness family F-03 found insufficient.** The transfer rests
     on the absence of F-03's three structural mechanisms. One residual mechanism, an orientation switch between plies,
     is named and logged (§2).
2. **Planes — six Markov planes, ten in-net line features.**
   - The planes: own, opponent, legal, fence, and stones-left as two planes.
   - The rivals' one-turn history planes are carded (§3).
   - The line features are parameter-free per-axis window counts and threat flags. A depthwise ±5 line conv sits in
     every block.
   - D6 is drawn per sample among the elements fitting the same square bucket. One orientation is served, chosen
     from the position alone.
3. **Trunk and heads — b10c96, hex-masked.**
   - 7-tap convs, masked BN, a KataGo pooling bias in blocks 3 and 7.
   - Heads: a per-cell policy plus `far`, dist65 value, run12's soft policy, and AUX-1's occupancy target.
   - Opponent policy and short-term value go to a later card.
   - **Cost:** ≈ 4.9 GF at the expected mean canvas (≈ 1 400 cells).
   - **GPU per leaf on the 4080S:** 0.19–0.6 ms. The span is the fused-versus-stock kernels and the shape split.
   - On a GPU-busy box this can slow the loop. L2 benches it against a ceiling, with smaller sizes as the fallback.
4. **Seam — a new representation `hexcanvas` (never `grid`).**
   - **New:** a registry row; an enum variant on both sides of the FFI (2 Rust and ≈ 23 Python dispatch sites); a
     canvas builder in `mantis-graph`; contract #12 `canvas wire v1`; a shape-tiered collate with CUDA graphs; a canvas
     sampler and trainer path; one arch dataclass.
   - **Shared:** search, tactics, the runner, the eval cache's order-free `LeafKey` with a representation word, the
     `LegalSetPolicy` reply, HEXG rings (after one reader rule), the ragged losses, dist65, the stamp and detector, the
     arena and the rulers.
5. **Bootstrap.**
   - **L3: a pretrain and a screen, not a capacity read** (R390(a)). Two seeds per architecture, dense b10c96 and GNN
     4×256, on R390(c)'s lab corpus with Six's labels excluded. Each architecture keeps its proven optimiser, both
     read with EMA.
   - **The screen is binary.** It STOPs only if dense − GNN < −0.17 on both rulers; otherwise the build proceeds.
   - **Controls:** the known-bad (permuted labels) must STOP. GNN seed 2 read against seed 1 must not STOP.
   - **Its rates, at seed terms of 0 to 0.10:** an equal design proceeds cleanly, with the positive control not
     voiding, 0.88–1.0 of the time; a design worse by 0.34 is STOPped 0.87–0.99.
   - **L4: does a dense run break run11's plateau within budget?**
     - This is the run question, by analogy with R391(a)'s width "read again only inside a run". It is not a
       controlled capacity verdict: no GNN runs beside it.
     - **The line:** one fixed four-cell window at a placement deadline must reach −0.42 logit on rung 16.
     - **A projection halt** stops a run that cannot reach the line. It halts 0.09 of on-pace linear runs and 0.75–0.93
       of runs that only reach run11's plateau.
     - **A miss is budget-bounded**, and says so.
6. **Cost.**
   - About 8 sessions to a self-play start: L0 0.5, L1 2, L2 2, L2b 1.5, L3 1, L4 1, plus reviews.
   - L3 takes ≈ 70–120 box-h of box A.
   - L4's deadline is 26M placements, Bubble's volume to its 185k release. That is ≈ 120–270 box-h at the dense
     loop's expected rate. Reaching run11's level is ≈ 50–160 box-h, and only if learning per placement matches
     Bubble's, which is unproven (§6.4).

Questions for the architect are in §9. The first asks whether a screen plus a plateau-break run is the read wanted,
since neither is a controlled capacity verdict.

## 1. The corrected premises, and the evidence in one table

| | Six (CixMango/Six f2b5ec2) | Bubble (Tomodovodoo/HeXO a408b85) | mantis GNN (d0a7346c) |
|---|---|---|---|
| frame | fixed 25×25 axial parallelogram, recentred per stone on floor(mean+½) of the last 4 stones; stones outside dropped; legal cells outside never searched except forced blocks (logit top − 8) | box of all stones + all legal cells (r 8) in the best axis pair; square buckets 24…256 in training, rectangles (multiples of 8, min 24) served; padding exactly inert (masked conv, norm, pool) | no frame: an axis graph over stones + legal cells (the fence = union of r-8 balls) |
| D6 | 12 drawn per sample, recomputed on coordinates; crop exact for 4 of 12 (within hex distance 12) | canonical orientation served; training draws among the 4/8/12 elements that fit the same bucket | per-record gate: 12 for a non-empty record, identity for the empty board |
| planes | 8: ones, own, opp, legal, this turn's first stone, opponent's last turn, second-stone flag, a dead radius-9 flag | 8: own, opp, legal, crop, remaining-1, remaining-2, this turn's first stone, opponent's previous turn | 11 node features (relative-7 + threat-4), 5 edge features |
| line features | LineWindows: parameter-free, input stage only, 6 binary planes (live 3/4/5 of one side) | LineFeatures: parameter-free, 10 planes (per-axis best open count, ≥4 and ≥5 flags) + a depthwise ±5 LineConv in every block | threat-4 node features from the builder |
| trunk | 7-tap hex-masked 3×3, BN, ReLU, pool bias in blocks 3 and 7; grown 10×96 (gen 10) → 10×128 (20) → 12×128 (40) → 15×128 (50) → 20×128 (140), 20×160 at gen 180 only, 20×128 to 455 | 7-tap hex-masked 3×3, masked BN, ReLU, pool bias every 3rd block; b6c96, never changed | GINE 4×128 (run11) / 4×256 (run12) |
| params | gen 30: 2.95M; gen 455: 5.91M | 1.10M | 4×128: 0.37M; 4×256: 1.13M (soft-policy kind) |
| GFLOP / position, served | gen 30: 3.67; gen 455: 7.36 (dense 3×3 as executed) | 1.19 at 24², 2.12 at 32², 2.82 training mean (1 365 cells) | 4×128: 0.23; 4×256: 0.73 (coded, B 64); training path 2.06 / 7.90 |
| value | 2 win/loss logits on z; a tanh score on Σπ′·completedQ the search never reads | one logit, BCE on z (value 1.5 + outcome 0.5) since 205k; the calibrated target dropped 10-08 by ablation | dist65 (65-bin two-hot on z) |
| aux heads (weights) | opponent policy 0.15; score 0.5; future occupancy at 6 and 20 stones 0.5 (on BCE summed over cells × 1/100); short-term value (λ 11/12) 0.5; rows weighted by policy surprise | opponent policy 0.15; future occupancy 3-class at +20 0.5; short-term value (λ 1 − 1/16) 0.5 (pair-policy 0.5 and proof-policy 0.25 are mixes into the policy target, not heads) | soft policy (aux 2.0); AUX-1's future occupancy landing |
| bootstrap | supervised v1 on HexBot alpha-beta + human games, then RL | from scratch (≈ 1 % rows converted from its relational run) | warm lineage run6 → run11; run12 from the Net2Net widening |
| serving | ORT 1.24.4 CUDA fp32, or TensorRT fp16; batches ≤ 32 in play, ≤ 1 024 in self-play | PyTorch eager + Triton fused line/norm kernels, cuDNN hex convs, bf16, channels-last, CUDA graph per (shape, batch), a planner that merges shapes | GINE with the fused Triton CSR op, bf16, CUDA graphs per bucket |

Bubble copied Six's planes and heads (its #105). So where the two agree, that is one design choice, not two.

## 2. The frame (Q1)

### 2.1 What our positions need

Every figure in this section comes from run11's self-play: 17 of its 153 hourly game shards, 2 285 936 positions,
read by the seam agent's read-only scripts (S4). The sample is run11 only, and its decision points sit a few plies
shallower than search leaves. L0 repeats the census over the whole lab corpus.

- **The legal set** is every empty cell within hex distance 8 of a stone: the GNN's fence, `gnn_axis_r8`. Its size
  has median 609, p90 852, p99 1 089, max 1 566.
- **Stone span along the widest of three axes:** median 17, p90 27, p99 36, p99.9 44, max 60. Along one axis it is
  median 14, p99 32–33.
- **The box that holds the whole legal set** is about span + 17 on a side: median ≈ 31, p99 ≈ 50.
- **What a fixed 25×25 crop centred on the bbox midpoint cuts:**
  - stones: lost from 11.9 % of positions;
  - legal cells: 23.6 % lie outside;
  - the played move: outside on 3.6 % of positions (3.2 % under Six's recent-4 centre).
- **On run11's final ring** (111 063 full-search rows), that crop leaves 2.5 % of explicit visit mass outside, and
  more than 10 % on 3.1 % of rows. A 32×32 crop leaves 0.36 %.
- **Six's own crop** leaves 32 % of our legal cells outside, on 85 % of rows (DISTILL-1).

### 2.2 The three candidates, judged

- **Six's fixed 25×25.** One shape, the cheapest to serve, and Six is strong with it.
  - It drops stones past hex distance 12 from a recency centre and scores no legal cell past it.
  - It is exact for only 4 of the 12 symmetries, and its trainer discards rows whose targets leave the crop.
  - Six leans on whole-board C++ tactics to cover what its net cannot see.
  - For us it would break arena legality's premise that every legal cell carries a prior. The played move would
    fall outside it on 3–4 % of our positions, and so would a quarter of the legal set. It brings back F-03's
    perception radius.

  **Rejected.**
- **A centroid-centred crop with an overflow rule.** Any centre is a frame. Under masking the centre stops mattering,
  and this becomes the next candidate with an arbitrary box. **Rejected as a separate design.**
- **Bubble's box with masked padding.**
  - No stone and no legal cell is ever outside in normal play, and its padding is exactly inert.
  - Bubble trained it from scratch and, within ≤ 44 h, beat Strix's public net 42–22 at budgets it did not publish.
  - **Adopted, with one change:** the mask is narrowed from the box to the fence. Bubble convolves the box's corners
    as real empty cells, and which corners exist depends on the orientation; narrowing buys exact D6. It is a choice
    with no evidence behind it either way. Bubble's full box is the fallback, a one-line arch field.

### 2.3 The fence canvas

- **Region and container.** Region = stones ∪ legal cells (radius 8). Container = the region's axial bounding box,
  in an axis pair (one of the three) chosen from the position alone (below). Each side is rounded up to a multiple of
  8, with a minimum of 24, which is the row's `canvas_min_side`. Served containers are rectangles; a lone stone's
  17 × 17 fence gets 24 × 24.
- **Masking.** Every op is masked to the region: conv inputs zeroed off-region, norm statistics and pools over region
  cells only. Zero outside the region is the truth, not a convention: no stone lies outside it and no cell outside it
  is legal.
- **Properties L1 pins.**
  - **Container invariance:** the output does not depend on the container's size or offset.
    - The tolerance is 1e-4 in fp32, a margin over Bubble's measured 2e-5 agreement. Bit-exactness is required only on
      a fixed-algorithm CPU path.
    - Served bf16 needs a bound across tiers too. Bubble measured worst-case policy-probability differences of 0.0028
      across paddings, so the bound is 5e-3 (R3).
  - **D6:** region(g·S) = g·region(S) for all 12 elements, so augmentation is lossless on every non-empty record. The
    empty board's 5×5 first-stone box is closed under 4 elements, so the per-record gate keeps it at identity, as for
    the graph today.
- **Training orientation (Bubble's rule, in square-bucket terms).** Training groups samples in square buckets, with
  the side the box's larger side rounded up to a multiple of 8.
  - Each sample draws uniformly among the D6 elements whose square bucket equals the served orientation's (4, 8 or 12
    of them, Bubble's `_choose`).
  - The training REGION mix then equals the served one, and every draw is exact.
  - The canvas mixes differ: square in training, rectangles and tiers in serving. The masking makes that difference
    output-neutral within §2.3's tolerances.
- **Serving orientation, from the position alone.** The rule picks the lowest-id axis pair whose square bucket is
  within one step (8) of the smallest. It is a function of the stone set, so the order-free cache key stays valid and
  training and serving agree. The band makes flips rarer than a strict minimum would.
- **The one residual frame mechanism.** Consecutive positions can still be served 60° or 120° apart through an
  untied, only learned-equivariant trunk.
  - It is logged in-run (LAW-18): the flip rate between consecutive plies on the played path, and the policy and
    value jump on flips against non-flips.
  - The pre-stated fallback is a fixed orientation: always the (q, r) axis pair. It is position-free and flip-free,
    at the cost of larger boxes for positions elongated along the third axis. A history-based hysteresis is ruled
    out: the output would stop being a function of the position, and the cache key would break.
- **Shape tiers.** Because masking makes padding exact, a leaf can be served in any container at least its own size.
  The collate merges each pop's shapes into a few tiers to keep batches large. That is Bubble's planner (#385/#413),
  and the trade (padding FLOPs against launches) is L2's to measure.
- **Far mode.**
  - `S_max` is a container side, in cells, chosen by L0 so that no lab-corpus position's fence box exceeds it.
  - Past `S_max`, the region shrinks to stones ∪ cells within 4 of a stone (Bubble's halo). Legal cells outside get
    `far − log n_far` each, and their target mass is kept in training.
  - `S_far = S_max + 32` bounds a far-mode root. A ROOT past it ends the game by the existing cap path, as in self-play
    (no z), and is scored as the cap's outcome in a cell. It is counted (LAW-18). L1 checks that the arena has such an
    outcome; if it does not, L1 designs one.
  - Leaves are never refused. A leaf can exceed its root by at most 8 cells per placement below the root, so L2's
    capture and memory budget covers `S_far` plus the search's depth margin.
  - Arena legality is untouched: every legal cell has a prior in every mode.

### 2.4 F-03 re-validated (LAW-02: the prior, its exact context, whether it transfers)

- **The row.** "Padding semantics (canvas-realness + partial convolution) recovers bbox direction". It was falsified
  by an ablation eval: "Trains below the anchor's loss yet 0 % SealBot WR; bbox direction is structural: K-aggregation
  as cross-cluster contrast, bbox-centroid frame instability, perception radius".
- **The code it tested is not in this tree.** `model/cnn.py` at its first commit, 313f1893 (2026-07-19), already reads
  "The v8 / PMA-pool / gpool-bias / canvas-realness branches of the old net are STRIPPED … (F-03/F-04/F-05/F-13)". The
  rows were curated from the predecessor project.
- **The context it names did survive, in the post-strip dense lineage** (3dd20b49^; S5):
  - axial windows of 19×19 (v6) or 25×25 (v6w25), each centred on its cluster's bounding-box midpoint, truncated
    toward zero (`board/state/cluster.rs:30-39`);
  - one window per action or threat anchor for wide clusters, up to K = 8 per leaf, with value = min and policy =
    scatter-max over windows;
  - square 3×3 convs with GN and SE;
  - a fully connected policy head, FC(2·H·W → H·W+1) (`model/cnn.py:4-14`).
- **Which mechanism failed.** By the row's own text, the padding fix was insufficient, not harmful: "bbox direction
  is structural" through three mechanisms.
  - (a) K windows contrasted by min/max.
  - (b) A frame whose centre moved with the bounding box, read by a fully connected policy head, so a one-cell
    recentre changed every logit's meaning.
  - (c) A window that could not see past its edge.

  F-13 adds that the padding-class collapse was "structural at K = 1 inference", in a single window. So (a) is not
  needed to explain it; (b) and (c) carry it. The register says no more, and the code that could is not in the tree.
- **What this design re-adopts.** Plane 3 (the fence) is a canvas-realness plane, and masked convolution belongs to the
  partial-convolution family, without renormalisation. Outside evidence that masking works: Bubble's masked canvas
  trains to strength (S2), and KataGo trains across board sizes with a 0–1 real-space mask applied after every op (S3).
- **Transfer.**
  - (a) is absent: one canvas per position.
  - (b) is absent: there is no centre, every op is translation-equivariant under masking, and the policy head is per
    cell. The one residual frame effect, the orientation switch, is named and logged (§2.3).
  - (c) is absent below `S_max`, and every stone is in the canvas in far mode.
  - **The falsified claim does not transfer.** It tested a padding fix layered on (a)–(c), and none of them is here.
- **Its instrument lesson transfers in full:** that net trained below the anchor's loss and lost every game. ENCODE-1
  and DISTILL-1 met the same in this tree, where held-out CE favoured the weaker net. No held-out loss gates strength
  anywhere below.

### 2.5 F-01 re-validated

- **The row.** "Hex-native trunk variant closes the self-play gap (2026-05-05)". It was falsified by MCTS-matched
  eval: "Probe gates passed but selfplay dropped to 0–1 % SealBot WR. Static probes cannot validate dynamic
  equivariance — only MCTS-matched eval can."
- **Its context.** The variant predates this repository, whose first commit is 5081ed1a (2026-07-16). Its code and
  record are not in the tree or its history; the row is all that survives. It ran inside the dense lineage of §2.4.
  F-09 places that lineage's own self-play collapse "at the argmax-degeneracy / selfplay-interaction layer, not
  corpus/loss", and F-22–F-28 record its colony attractor.
- **Which mechanism failed.** The register names an instrument failure, not a trunk mechanism: a net that passed
  static probes collapsed in self-play. "Dynamic equivariance" suggests a trunk built to be equivariant. That is an
  inference; the record that would settle it is not in the tree.
- **Transfer.**
  - The causal reading, "a hex-native trunk collapses in self-play", is contradicted outside our tree. Six trained a
    7-tap hex-masked trunk by self-play from gen 10 to gen 455, and gen 455 at 16 nodes per turn wins about 64 %
    against run11's plateau at 128 per stone (rung 16, −0.589 logit). Bubble trained the same kernel from scratch.
  - Both use untied kernels and augmentation, not built-in equivariance, and so does this design.
  - The F-01 collapse cannot be separated from its lineage's own collapse.
- **The instrument lesson transfers in full.** No static probe gates strength: not the orientation spread, held-out
  CE, golden parity or the conformance suite. Those pins gate correctness (L1, L2). Strength is read only on
  MCTS-matched panels: L3's screen and L4's run line, with its halt (§6).

### 2.6 The other rows this design touches (LAW-05)

- **F-38: "the deeper-net frontier is NOT earned. The lever is search-in-the-loop".**
  - Context (2026-06-26): the 33 value-blind proven-core losses of that era were all bounded-tactical, proven at depth
    6–8 turns.
  - What has changed since:
    - The lever F-38 named is live at deploy (R377/R378, about +0.96 logit).
    - It is also live in run11's self-play: arm A's design was carried into run11, not adopted, after two positive
      screens over plain, +0.65 and +0.17 (R381(c)).
    - run11 plateaued anyway, and R391(c) names "a thin representation signal … capacity unused at a floor LR" as
      causes of record.
    - F-38 falsified "a deeper net is needed for those 33 losses", not "capacity never limits".
  - It transfers as a warning: DENSE-1's premise is the class F-38 found unearned once. So nothing here assumes
    capacity is the limiter. L4 reads it inside a run, and a miss halts.
- **F-36: "Richer input threat planes … the bottleneck is not cheap input features".** Transfers. The ten line
  features are adopted as a cheap inductive aid that both rivals use, never as a lever or a reason to restart.
- **F-05: "Policy-only gpool-bias is load-bearing; full gpool-bias is NULL on value".** The trunk-level pooling bias
  (KataGo's 1.60×; Six; Bubble) is kept for policy. F-05 predicts no value gain from it, and none is claimed.
- **F-33: "No pre-selfplay bias exists — the attractor is GENERATED by the training loop".** Transfers to L4. A clean
  pretrain does not protect the run, so L4 carries the in-run monitors and the halt.
- **F-04** (PMA versus min/max K pooling): there is no K here.
- **F-07** (more pretrain epochs regressed self-play; the value head over-fits corpus signal): partly transfers. The
  pretrain budget is fixed in advance and never stopped on held-out loss, and the value keeps run12's 1/8 mask.
- **F-08 / F-09** (the collapse was not radius or bootstrap): nothing here re-tests either.
- **F-11** (fp16 aux CE NaN): bf16 under LAW-06, aux CE in fp32.
- **F-14** (8 of 18 planes suffice): consistent with 6.
- **F-21** (no custom kernel for the dense path) does not transfer. Its premise was a dense path that "varies batch
  COUNT, not tensor SHAPE", and the canvas varies shape. The serving order "stock path first, measured, then fused
  kernels" stands on Bubble's measured sequence (S2) instead.

## 3. The planes (Q2)

| # | plane | definition |
|---|---|---|
| 0 | own | the mover's stones |
| 1 | opponent | the other side's stones |
| 2 | legal | legal cells (the policy's support) |
| 3 | fence | the region (own ∪ opp ∪ legal below `S_max`; the far region in far mode); also the mask |
| 4 | stones-left 1 | broadcast 1 inside the fence when one placement is left this turn (the second stone, or ply 0) |
| 5 | stones-left 2 | broadcast 1 inside the fence when two are left |

- **Side to move:** only through the own/opponent split, as in both rivals. Planes 4–5 are `moves_remaining` one-hot,
  as in Bubble. Every plane is a function of what the eval cache key already holds: stones, side, moves left.
- **History (CARD-DENSE-HISTORY), not in the first row.** Six adds two planes (this turn's first stone; the
  opponent's previous turn as a set), and Bubble copied them. Neither ablated them. The real costs:
  - **a key word.** `LeafKey` is an order-free XOR, pinned by `one_position_in_any_stone_order_is_one_key` (S4). A
    history row needs a word for the previous turn's stones (an unordered set) and this turn's first stone. That keeps
    the common within-turn hit: a repeat evaluation, where the second stone's search re-reaches what the first stone's
    search already evaluated (39.9 % of the second stone's evaluations, SEARCH_PERF §2.4). It loses only cross-turn
    transpositions that touch the last turn. That fraction of the 32 % hit rate is unmeasured; L0 can count it from
    search stats where they exist;
  - **a ring field.** It stores the previous ≤ 3 placements the way AUX-1 stores the next 20 (S6);
  - **exposure.** A played leaf's `Board` already keeps `action_anchors`, the last 4 stones (`pub(crate)`). A board
    rebuilt from a stone set has none, which the sampler path must handle.

  Played reads need all of this before history can be read, and a held-out read is no verdict (F-03). So the first
  row is Markov, and the card opens after L3. Its read is a pair, Markov against history, on the same screen.
- **Line features, computed in the net, parameter-free, on own and opponent:**
  - for each side and each of the 3 axes, the best stone count over the 6-windows through the cell that hold no enemy
    stone, ÷ 6 (6 planes). Every stone lies inside the fence, so windows crossing the fence edge read exactly. There is
    no in-crop restriction, unlike Bubble's;
  - on empty fence cells, own ≥ 4, opponent ≥ 4, own ≥ 5 and opponent ≥ 5 (4 flags), each thresholding the max over
    axes;
  - the stem input is 6 + 10 = 16 channels.
- **Line features against the alternatives.**
  - Six's set marks cells in live windows of exactly 3, 4 or 5. Bubble's graded max does not strictly contain it; it
    is the default because per-axis counts keep axis identity.
  - Our graph's threat-4 node features are not reused: that would tie the canvas builder to the graph builder's
    feature code. The CPU builder stays at six trivial planes (Six's argument, S1 Q3).
  - Precomputed threat planes are not used either, by F-36 and because in-net features cannot diverge between
    trainer and server.

## 4. The trunk and the heads (Q3)

### 4.1 Trunk

- **Stem:** HexConv(16 → C) on the masked input.
- **Block (pre-activation):**
  - `y = HexConv(ReLU(MaskedBN(x)))`;
  - in blocks 3 and 7, add `Linear(3C → C)` of the masked pools of `ReLU(y)`: mean, mean scaled linearly with the
    canvas width (√ fence cells), and max. That is KataGo's pooling bias, which scales by board width because its
    board size varies, as the canvas does here (S3);
  - `y += LineConv(y)`, a depthwise conv of length 11 (±5) along each of the 3 axes, summed, zero-initialised;
  - `x + HexConv(ReLU(MaskedBN(y)))`.
- **HexConv:** a 3×3 conv with taps (−1,−1) and (+1,+1) fixed at zero, leaving the centre and 6 untied hex neighbours.
  cuDNN computes all 9 taps, so the mask saves no FLOPs (S1, S2).
- **MaskedBN:** batch statistics over fence cells only, reduced in fp32. Running statistics are recomputed over a
  fixed panel at each save (Bubble's fix for its early EMA-statistics bug, S2).
  - **Training grouping:** square buckets (sides that are multiples of 8), each padded to a quantum of 4 rows
    (Bubble's). A group's loss is weighted by its share of the step's denominator.
  - **Serving** uses running statistics, so no output depends on its batch-mates through the norm.
  - **Re-measured in L2 (R3):** cuDNN's per-batch kernel choice, against the eval cache's precondition (R370(c): a
    replay inside the served path's own batch-size spread).
  - **Pre-stated fallback:** a per-sample channel norm, which has no batch statistics, if L2's parity pin fails.
- **Size: b10c96.** 1.82M parameters stored, 1.45M effective without the masked taps. For comparison, Bubble's b6c96
  has 1.10M and our 4×256 GNN 1.13M with its aux head.
  - It is Six's starting net (gen 10), between Bubble's b6c96 and Six's gen-30 10×128.
  - KataGo's first run started at b6c96 and moved to b10c128 at about 0.75 days (S3).
  - **Growth (L5)** is function-preserving, Six's way: new channels get zero outgoing weights, new blocks a zeroed
    second norm. The order is 10×96 → 10×128 → 15×128, and a pre-stated plateau test fires it.
    `mantis.model.widen` is graph-only: its rule-per-tensor pattern carries over, its code does not.

### 4.2 Heads

| head | form | target | weight |
|---|---|---|---|
| policy | per cell: 1×1 C→32, ReLU, 1×1 32→1, over legal cells; `far` = Linear(3C→1) on the pooled trunk, far mode only | run12's improved policy (full-search rows) | 1 |
| value | masked pools → Linear(3C→128) → ReLU → dist65 (65 bins) | z under the 1/8 mask (run12) | run12's |
| soft policy | as run12's aux head, per cell | run12's soft target | run12's aux 2.0 |
| future occupancy | per cell 1×1 → 3 classes {empty, own, opp} at +20 placements, zero weights and a quiet bias | AUX-1's ring v4 field, labelled over the reader's legal set | AUX-1's minted weight, if AUX-1 lands |

- **Value stays dist65,** so the L3 pair differs in architecture only. The rivals' one- and two-logit heads are no
  evidence against bins: neither compared.
- **Occupancy weights are not comparable across the rivals.** Six's 0.5 multiplies a BCE summed over cells × 1/100;
  Bubble's 0.5 multiplies a per-cell 3-class CE. AUX-1 mints its own.
- **CARD-DENSE-AUX, a later card:**
  - opponent policy, 0.15 in KataGo, Six and Bubble. KataGo's ablation credits it 1.30× (S3). Six's target form, the
    opponent's next-turn cells, comes free from AUX-1's v4 field;
  - short-term value, 0.5 in Six and Bubble, λ = 1 − 1/12. It needs later rows' root values: offline through the game
    records, or in self-play through one more ring field;
  - Bubble copied both from Six, so they are one choice. They join the dense kind only on a powered read of their own
    on the supervised harness, after L3.
- **No score head.** Hex Tac Toe has no score. Six's "score" is Σπ′·completedQ, the short-term value's own input, and
  Six's search never reads it.

### 4.3 FLOPs per position (FLOP = 2 × MAC, dense 3×3 as executed; `design/flops.py`)

| net (our stem and heads) | 24² (576 cells) | 32² (1 024) | mean ≈ 1 400 | 56² (3 136, ≈ p99) | marginal per cell |
|---|---|---|---|---|---|
| b6c96 | 1.22 | 2.16 | 2.96 | 6.62 | 2.11 M |
| **b10c96 (proposed)** | **2.00** | **3.55** | **4.85** | **10.9** | **3.46 M** |
| b10c128 (Six gen 30's shape) | 3.51 | 6.24 | 8.53 | 19.1 | 6.09 M |
| b20c128 (Six gen 455's shape) | 6.96 | 12.4 | 16.9 | 37.9 | 12.1 M |

| GNN (S4) | served, coded edges, B 64 | training path |
|---|---|---|
| 4×128 (run11) | 0.21–0.28 | 2.06 (edge linear 1.75) |
| 4×256 (run12) | 0.70–0.90 | 7.90 |

- **The mean canvas.** The fence's box is about span + 17 on a side. Bubble measured its training mean under the
  same rule at 1 365–1 471 cells (S2). Our run11 spans put the median near 32² and p99 near 56². L0 measures ours.
- **Bubble's own net** with its own stem and heads is 1.19 / 2.12 GF at 24² / 32².
- **In FLOPs only:** b10c96 at the mean canvas is ≈ 7× the 4×256 GNN served, and its forward is ≈ 0.6× the GNN's
  training-path forward.
- **FLOPs are the wrong unit across these architectures.** A scatter-gather GNN and a tensor-core convolution run at
  very different efficiency, so the design sizes by measured time per leaf on the 4080S (§4.4).
- **Six-class capacity does not fit our served FLOPs.** Six gen 30 at its own 625 cells is 3.67 GF, 5× the 4×256
  served.

### 4.4 Per-leaf cost on the 4080S (estimate; L2 measures it)

- **The anchor.** Bubble's measured backend ran b6c96 at 24², batch 32, at 8 897 rows/s on a 3070 Ti. That is
  10.6 TFLOP/s achieved, with CUDA graphs and fused Triton line and norm kernels, at one shape (S2).
- **The GPU factor.** The 4080S is 2.40× the 3070 Ti by FP32 spec (10 240 cores × 2 × 2.55 GHz ≈ 52.2 TFLOPS,
  against 6 144 × 2 × 1.77 ≈ 21.7; S3). Our own GNN trainer step ran 1.9× faster on the 4080S than on the 3070 (not
  the Ti), scatter-bound, beside cells (S4).
- **The fused-kernel factor.** L2 serves the stock path first. Bubble credits its fused kernels plus CUDA graphs with
  2.76× on actors (S2); we have CUDA graphs.
- **The shape-split factor.** run11 pops hold about 36 leaves (S4). Mixed rectangles split a pop into small groups,
  unless the tier merge (§2.3) keeps the groups large.
- **b10c96 at the mean canvas** (4.85 GF):

  | path | GPU per leaf | leaves/s |
  |---|---|---|
  | fused, merged | 0.19–0.24 ms | ≈ 4 100–5 200 |
  | stock, split | up to ≈ 0.6 ms | |
  | b6c96, fused | 0.12–0.15 ms | |

- **The GNN today on box A.** The 4×128's real B-64 forward takes 6.5 ms with the edge table, ≈ 0.10 ms per leaf.
  `bench_server` serves 3 984–5 295 leaves/s end to end, and the trainer-free loop ran 4 505 leaves/s at 89 % GPU.
  The 4×256 served on box A beside run11 at +7.6 % µs per leaf over the 4×128 at B 1 and 8 (DISTILL-2). At the
  serving batch size the desktop read it at +27.5 % at B 64 (941 against 738 µs, DISTILL-1).
- **CPU.** The graph build costs 622 µs per leaf in-run, plus ≈ 83 µs of pack and semantic checks. The canvas build
  is a scatter of six planes.
- **What it means.** The dense kind trades CPU for GPU, on a box whose GPU is already 89 % busy without a trainer.
  More GPU per leaf slows the loop unless the CPU saving is the binding term. That is the likeliest place for this
  design to fail.
- **Falsifier: L2's bench,** pre-registered with its bracket and abort threshold (LAW-09), on the stock path first.
  - Ceiling: `bench_server` leaves/s at L0's shape mix ≥ 0.8 × the 4×256 GNN's, both measured on box A in the same
    session.
  - Below it, in order: fused kernels, b8c96, b6c96, then the margin option (§9 Q3).

## 5. The seam (Q4)

All sites were read at d0a7346c. S4 (the seam report) has every `path:line`.

### 5.1 What exists, and what is closed today

- **The old dense path.** It existed and was deleted by R346(f); tag `archive/grid-path` keeps it.
  - `Representation::parse` refuses `"grid"` by name, and the Rust enum has one member, `Graph`
    (`crates/mantis-encoding/src/spec/mod.rs:33-51`).
  - Eight tests pin "grid" and "dense" as refused tokens, among them `registry_census.rs:65-68` and
    `test_legal_move_coverage_boundary.py:185-188`.
  - `HexTacToeNet` is a pinned grave. The report-only detector reads `representation.input_proj.weight` as the graph
    marker.
  - **Hence:** the new kind is named **`hexcanvas`**, its net is not called `HexTacToeNet`, and no module is named
    `representation.input_proj`.
- **Dispatch.** Rust has two exhaustive `match` sites: `spec/validate.rs:168`, and `resolve_geometry`
  (`runner/params.rs:78`), which the runner, `HexgBuffer::new` and the bridge batcher call.
  - Python has about 20 sites in `src/` and 3 in `tools/`.
  - Two are easy to miss: the literal `amp_dtype_for("graph")` (`inference_server.py:425`), and
    `SOFT_POLICY_ARCH_KINDS` (`config/schema/core.py:35`).
  - The trainer step dispatches through `run_declared_train_step` (`train/coordinator/dispatch.py:116-123`).
- **Registry relics.** These would bind a `hexcanvas` row by accident:
  - `sym_table_id` ∈ {size_19, size_25}, where size_25 needs `n_planes == 8`;
  - `n_chain_planes == 6`;
  - `empty_edges`, required on every row.

  The schema states a rule for each.

### 5.2 The registry row

One new row, `hexcanvas_r8`, with the schema version bumped:
- `representation = "hexcanvas"`;
- the reply frame exactly as the graph rows carry it: `board_size` 19, `trunk_size` 19, `policy_logit_count` 362,
  `sym_table_id` size_19. These describe `LegalSetPolicy`'s 19-window and overflow map, which stays the one
  search-facing reply;
- `legal_move_radius` 8, `win_length` 6;
- `n_planes` 6 with its `plane_layout`;
- `canvas_side_step` 8, `canvas_min_side` 24, `canvas_max_side` (`S_max`), `far_halo` 4, `far_max_side` (`S_far`);
- no graph-only keys, and `empty_edges` refused for this representation.

The line features belong to the arch: the row describes the wire. The census adds the row and its field pins; LAW-08's
live-consumer check is the census (R389(a)).

### 5.3 What is new, and what is shared

| layer | shared as is | new for `hexcanvas` |
|---|---|---|
| rules, legal set, search, tactics, deploy block | all | none |
| leaf selection, expand, backup | all; expand reads its frame off the spec | none |
| eval cache | `LeafKey`'s order-free XOR of stones, side and moves left, which covers the Markov planes exactly | a distinct representation tag beside the geometry word (the graph rows keep their edge-mode word), so graph and canvas entries never collide; `the_key_separates_every_input_the_builder_reads` extends to it |
| input build | `mantis-graph`'s legal set, `BallGrid`, `StoneIndex` | `build_canvas` in `mantis-graph` (std-only, wasm32-clean, gate 4): fence, orientation, container, 6 planes, legal gather list and reply slots |
| queue, wire | the reply `(LegalSetPolicy, value)` | `PendingCanvasRequest`; contract #12 `canvas wire v1`: `[B, 6, H, W]` per shape tier, the mask, per-leaf legal gather indices and `policy_dst_slot`s |
| collate, checks | the 1-in-1 period discipline | the tier planner (§2.3); checks: legal cells inside the fence, gather and slot agree, mask = region, each with a mutation self-test (LAW-07) |
| forward | bf16 (LAW-06), the CUDA-graph capture mechanism (`BucketedForward`) | one captured graph per (tier, batch capacity) under a capture-memory cap; masked-norm statistics in fp32; `resolve_amp_dtype` → bf16 |
| decode | `submit_graph_inference_results` → `assemble_ls_from_gnn_probs` | the net gathers legal-cell logits in the wire's legal order; `far` cells get `far − log n_far` |
| ring | HEXG rows: stones, coordinate-keyed visits over the full r8 legal set, z, flags, root value; AUX-1's v4 field | `reads_rings_of` admits a `hexcanvas` reader of a graph row with equal radius and win length; `resolve_geometry` admits the representation. One ring format stays |
| sampler, trainer | D6 coordinate rotation (`replay/sym.rs`), the ragged-over-legal losses, dist65, soft policy, the trainer loop and its protected guards | the canvas sampler (rotate, build, scatter targets); bucketed microbatches (§4.1); `run_declared_train_step` for `hexcanvas`; the save-time BN-statistics recompute; `train.microbatch_caps` in cells; the sampler reads the HEXG weight column, stored today at 1.0 and read by none, for lineage weights |
| arch, config, stamp | `load_checkpoint` / `resume_trainer` with `declared_encoding`, lineage, content hash, the strip gate (which refuses dense↔graph already) | `HexCanvasArch` in `ARCH_KINDS`, `ARCH_KINDS_BY_REPRESENTATION`, `INCUMBENT_ARCH_KIND`; a `model.hexcanvas` block in `ARCH_SCOPED_KEYS`; `_SYNTH_ARCH_SCOPED` rows; a width-drift check |
| arena, rulers, cells, dash | all; a player is built from the stamp | none |

### 5.4 The conformance suite

- **Met as written:**
  - the roster guard;
  - no second kind table;
  - no symmetry-named arch field (so `HexCanvasArch` carries no `d6_*` field);
  - graves and reachability;
  - the 11 search-kind tests, which need full legal coverage, and the fence gives it.
- **New entries.** These four are built on graph objects, so each needs a `hexcanvas` entry pinned like its graph
  twin:
  - the memory envelope;
  - the perf floor (both hand-list the Gnn kinds by `startswith("GnnArch")`);
  - the selector round trip to a served batch;
  - the throughput-harness forward.
- **Explicit changes:**
  - **T9 config partition:** `REPRESENTATIONS = ("graph", "hexcanvas")`, and the block joins `ARCH_SCOPED_KEYS`.
  - **T4 legal coverage:** the archived two-sided dense boundary arm is re-derived for the fence, and its three unused
    refusal classes come back into use.
  - **T1 window frame:** unchanged. The canvas has no centre, and its origin is the region's minimum corner.
  - **Widening:** a rule table of its own (L5).
- **Protected pins that must hold for `hexcanvas`, parametrised over every registered row:**
  - served-sims exactness;
  - the conformance sections;
  - the 1-in-1 collate checks;
  - arena legality;
  - the finite-gradient guard;
  - the resume round trip.

### 5.5 Repo-design, contract and map amendments (L1, before code)

1. **repo_design §3:** the representation reads `"graph" | "hexcanvas"`, and `"grid"` stays refused by name.
2. **The R346(f) amendment, annotated:** the dense path that died was the K-window grid. `hexcanvas` is a new
   representation with a new wire, and nothing of `archive/grid-path` is restored.
3. **repo_design §2 and CLAUDE.md's map:** `mantis-graph` becomes the axis-graph and hex-canvas builders.
4. **Contracts:**
   - **#12 `canvas wire v1`.** #2 and #8 stay retired: a number is retired, not reused.
   - **#1, the registry:** the new representation and keys.
   - **#6, replay persist:** `reads_rings_of` across representations, pinned at `registry_census.rs:225-237`.

## 6. The bootstrap (Q5)

### 6.1 Corpus and targets

- **The corpus is R390(c)'s lab corpus:** 491 392 deduplicated self-play games, eval and benchmark run ids excluded,
  lineage-tagged by run.
  - It holds ≈ 40.5M positions with z and 8.13M full-search ring rows, once deduplicated by content (the ring
    `game_id` resets on resume).
  - Those two counts come from the 2026-10-08 census. Its only record is a session note, so L0 re-derives them into
    the records first.
  - Rows are HEXG, read by the canvas sampler as they are.
- **Exclusions and joins.**
  - Six's labels are excluded (R391(e)), so DISTILL-1's labelled set never enters. Neither arm is trained on a
    ruler's outputs (R391(a)).
  - AUX-1's occupancy field for corpus rows comes from their game records, joined at 99.99 % by
    `tools/distill/corpus.py`.
- **Targets** (run12's): policy CE on full-search rows; dist65 value on z under the 1/8 mask; soft policy; D6 per
  §2.3.
- **Lineage weights.** run11 and its fresh line carry full weight, older runs are down-weighted, and the schedule is
  pre-stated (§9 Q5).

### 6.2 L3 — the pretrain and its screen (not a capacity read)

**Why a screen.** R390(a) says "from-scratch arms at feasible budgets read budget, not capacity". So L3 asks only
whether the dense pretrain is broken or clearly worse than an equal-budget GNN pretrain, before a run is spent on it.

**The arms.**
- Each arm runs 64k steps at B 256, two passes over the 8.13M full-search rows, saving every 4k.
- Each panel is its last four saves; the known-bad's panel is its last save.
- The arms are:
  - dense ×2 (b10c96);
  - GNN ×2 (4×256, run12's shape);
  - known-bad ×1: dense on permuted policy labels.

**Recipe, stated as a confound.**
- Both architectures share run12's targets and mask.
- Each keeps its proven optimiser. The GNN runs run12's cosine 1e-3 → 1e-4. The dense kind runs Bubble's AdamW at
  3e-4 after warmup, the only dense recipe proven in this game.
- **Both are read through an EMA of 0.999** (`mantis.train.ema`). Bubble's audit puts raw weights 50–100 Elo below
  EMA, so EMA on one side only would bias a non-inferiority screen.
- The screen therefore reads architecture plus optimiser, and says so.

**The rule (binary, pre-stated).**
- **STOP:** the mean dense panel minus the mean GNN panel is below −0.17 on rung 16 AND on S.
  - The build returns to the architect. Under R390(a) a STOP is not a falsification of the kind.
- **PROCEED:** anything else. A one-ruler deficit is reported as MIXED and proceeds.
- No other exit exists: the budget is pre-stated, and the screen claims no capacity read. Each arm's last-four-save
  slope is reported for information only.

**Controls (LAW-19).**
- **Known-bad:** it must STOP against the GNN pair. DISTILL-1's equivalent read 0.028 on rung 16, 1.63 logit under
  its control.
- **Positive control:** GNN seed 2, as the candidate, against GNN seed 1, as the control, must not STOP. If it STOPs,
  the seed term is too large for the line, and the screen is VOID and goes to the architect.

**Witnesses.** These guard breakage, never strength.
- At the panel saves (the last four), each arm reads the calibrated exams and the GEN instrument (held-out value AUC).
- Floors sit halfway between DISTILL-1's known-bad and its 10k scratch control:
  - value AUC ≥ 0.62 (0.52 / 0.727);
  - T4_V ≥ 0.09 (≈ 0 / 0.183);
  - DEF_V_att ≥ 0.16 (≈ 0 / 0.328).
- An arm under a floor is reported as suspect. If its panel reads strong, the architect rules: a held-out probe never
  overrides a played read (§2.5).

**Power** (`design/power.py`, Monte Carlo; `design/power.out`).
- **Operating point:** scratch arms sit near rung 16 p ≈ 0.15, where 576-game cell noise is 0.117 logit. S is
  assumed near p ≈ 0.30, unmeasured for scratch arms.
- **Scatter:** the save scatter is R391(c)'s non-noise term, ≈ 0.09. The seed term is one per arm, shared across
  both rulers.

| seed term | equal designs STOP | positive control does not STOP | clean PROCEED, equal designs | worse by 0.34 STOP | known-bad STOP |
|---|---|---|---|---|---|
| 0 | 0.000 | 0.999 | 0.999 | 0.986 | 1.000 |
| 0.05 | 0.003 | 0.984 | 0.982 | 0.956 | 1.000 |
| 0.10 | 0.036 | 0.916 | 0.883 | 0.872 | 1.000 |

- One seed per architecture STOPs a design worse by 0.34 only 0.92 / 0.86 / 0.76 of the time at the same seed terms
  (`design/one_seed.py`). Hence two, the count pre-stated (R381(b): multi-seed twins).
- No clean between-seed measurement is on file, so these rows are the power line.
- The minimal effect the screen claims to stop is −0.34, twice the line (§9 Q2). Smaller deficits are left to L4.

### 6.3 L4 — does a dense run break run11's plateau within budget? (only on R391(d)'s trigger and L3's PROCEED)

**What it is.** The run question, asked the way R391 asks it of run12: against run11's plateau. Width, by R391(a),
is "read again only inside a run", and this applies that by analogy.
- It is **not** a controlled capacity verdict: no GNN runs beside it, so a pass does not separate the architecture
  from the pretrain, the recipe or the hours.
- A miss at the deadline is budget-bounded, and the exit says so (R390(a)).
- A controlled verdict would need an equal-recipe GNN run beside it, twice the box (§9 Q1).

**The run.**
- It starts from the dense seed with the higher mean panel across the two rulers.
- It carries the dense optimiser from L3: AdamW at 3e-4, EMA 0.999, and a pre-stated drop to 1.5e-4 at half the
  deadline. That is Bubble's two settings.
- Everything else is run12's: 2.5M window, 48 workers, 2.4 steps per game, mask 1/8, the arena gate book, the deploy
  block of record.

**The deadline: 26M self-play placements.**
- That is Bubble's volume to its 185k release: 104 h × ≈ 250k/h.
- It is converted to steps at the preflight's measured placements per step. That is ≈ 720k steps, or 30 cells of 24k,
  at run11's ≈ 36 placements per step.
- In box-hours it is ≈ 120–270 h, at the dense loop's expected 0.6–1.0 × run11's placement rate (§4.4).

**Cells.** Rung 16 every 24k, and S every 48k, 576 games each, under DEPLOY-2's cap.

**The line, read once in a fixed window: the last four rung-16 cells before the deadline.**
- Their mean must be ≥ −0.42 logit, run11's plateau of −0.589 plus 0.17.
- The two S cells in that window must have a mean no more than 0.17 below run11's S panel. That panel is the mean of
  run11's S cells over its rung-16 plateau window, 228k to its stop, computed from the cell sidecars at L4's
  PRESTATED.
- At zero effect (a run that only reaches the plateau), the window passes with probability 0.000–0.002. At a run on
  pace for +0.34 it passes 0.90–0.98 (`power.out`).
- These rates are for rung 16 only. The S condition lowers them by about 0.03.
- The model draws cell noise independently. Correlated save-to-save noise along a run would widen both the window and
  the halt.

**The futility halt (a projection, not a slope threshold).**
- Looks come at cells 10, 14, 18, 22 and 26.
- At each look, an OLS fit over the last 10 rung-16 cells is projected to the window's centre at the slope's upper
  bound, slope + 2.31 SE (t₈, one-sided 97.5 %). If that projection is below the line, the run halts.
- Its rates, under a per-cell SD of 0.14 and a start 1.2 logit under the plateau:

  | true run | halted |
  |---|---|
  | on pace for +0.34, linear | 0.085 |
  | on pace for +0.34, concave (front-loaded) | 0.009 |
  | only reaches the plateau | 0.75–0.93 |
  | flat | 1.00 |

- **Its stated limit:** a run that accelerates late (convex) is halted 0.96. No projection rule can foresee
  acceleration. Learning curves here have been concave, but that is a judgement.

**Other halts.** run12's exams, bands and gap rule.

**Which weights play.** The actors and the cells play the EMA weights, as Bubble's exports do. That arms `train.ema`,
which the gate, the anchor, the follower and warm starts also read (FORGE-2). The PRESTATED states it.

**In-run rows (LAW-18):**
- the far-mode rate and refusals;
- orientation flips, and the jump on flips;
- uncaptured shapes.

### 6.4 Hours to run11's level (inference, not a claim)

- **Bubble.** From scratch it beat Strix's public net 42–22 in ≤ 44 h, and reached its 185k release in 101–104 h, on
  one shared 3070 Ti (S2).
  - Between 125k and 185k it retained ≈ 156k rows/h. At its own retention of 0.625 per placement, that is ≈ 250k
    placements/h (inference).
- **run11 on box A** made ≈ 162k placements/h, and ≈ 195–214k after cycle 2, at ≈ 73 GPU evaluations per placement
  (3 285 evals/s ÷ ≈ 45 placements/s; S4).
- **The bracket.** At parity of learning per placement, and with the dense loop at run11's placement rate (L2
  decides), Bubble's 44 h and 104 h scale by 250 / (162–214) to ≈ 51–68 h and ≈ 122–160 h. Hence ≈ 50–160 box-h.
  - At Bubble's later hybrid-era rate (≈ 158k placements/h), the low end falls to ≈ 32 h.
  - At the dense loop's possible 0.6× run11's rate, the top rises to ≈ 270 h.
  - That is why L4's deadline is set in placements, not hours.
- **What the bracket does not know:**
  - parity per placement is unproven. Our placements carry more search, and its targets come from a different
    search;
  - no Bubble milestone maps to rung 16. The nearest anchor is the Strix axis: Bubble at 85k scored 0.66 against
    Strix's public net, at unpublished budgets, while run11's best read 0.628 against our S pin, budget-matched.
    Different Strix nets and budgets make that an order-of-magnitude anchor at most;
  - the pretrain's head start is unknown. DISTILL-1's 15k scratch control sat 1.1–1.35 logit under the plateau
    (panel −1.93 against −0.589), and a 64k pretrain should start higher by an unmeasured amount.
- **For scale:** run11 took ≈ 43 h from 32k to the start of its 228k plateau, at 4 508 steps/h, and that run was
  warm.
- L4's deadline is the bracket's top, in placements: 26M.

## 7. Risks, each with its falsifier

| # | risk | falsifier (where) |
|---|---|---|
| R1 | A dense run does not break run11's plateau within its budget. Its causes cannot be told apart from L4 alone: the architecture, the compute, the pretrain or the recipe. R391(c) names the window, a thin representation signal and a floor LR; DISTILL-2 read the width "unused at lr 1e-4 in 15k steps"; F-38 found depth unearned once. | L4: the fixed window at the placement deadline, or the futility halt. A miss is budget-bounded. L3 catches only a broken or clearly worse pretrain. |
| R2 | The fence canvas costs too much GPU on a GPU-busy box. | L2's bench ceiling (§4.4), then fused kernels, b8c96, b6c96 and the margin option, in order, decided before L3. |
| R3 | A masked-norm train/serve mismatch, or serving that varies with batch or tier beyond the cache's precondition. | L2 pins: the served output equals the trainer's eval-mode output on a golden batch within tolerance; the bf16 served output varies by ≤ 5e-3 in policy probability across tiers and batch sizes; R370(c)'s cache precondition is extended to that spread. |
| R4 | Variable shapes defeat CUDA graphs or blow the capture memory. | The L2 bench at L0's shape mix: capture count and memory under the cap; uncaptured shapes counted (LAW-18). |
| R5 | The encoding is not D6-exact or not container-invariant (a bug). | L1 tests: 12 elements × N positions give an exactly re-indexed input; container size and offset move fp32 outputs by ≤ 1e-4; bit-exact on a fixed-algorithm CPU path. |
| R6 | The orientation switch between plies destabilises search (§2.3). | L4's in-run row: the flip rate and the jump on flips against non-flips. The pre-stated fallback is the fixed (q, r) orientation. |
| R7 | Bootstrap bias: weak old targets, value over-fit (F-07), or a lineage mix that over-weights run6–run8. | The pre-stated lineage weights, played panels only, and L3's witnesses. |
| R8 | The Markov row leaves strength on the table that the rivals' history planes would give. | CARD-DENSE-HISTORY after L3: Markov against history on the same screen. |
| R9 | Far mode or refusals fire in play beyond the corpus. | LAW-18 counters in-run; `S_max` and `S_far` from L0 with a stated margin. |
| R10 | run12 climbs, and DENSE-1 never builds. | R391(d)/(e). This document is shelf-ready; only L0 could run before the trigger, and only on a ruling. |

## 8. The build plan — legs, gates, cost

Each leg ends with `make gates`, a fresh read-only review agent with its findings fixed (CLAUDE.md's review gate),
and the operator's word to push. Strength is read only on panels.

| leg | what | gate | sessions |
|---|---|---|---|
| L0 CENSUS | Records only. Over the lab corpus and run11's final ring: fence box sides under the position-only orientation rule (§2.3, which can serve a box one bucket over the minimum), the shape mix, `S_max`; visit mass by hex distance to the nearest stone (for §9 Q3); within-turn against cross-turn cache transpositions where search stats exist; and a re-derivation of the corpus counts. | Pre-stated before reading: `S_max` = the smallest multiple of 8 above every corpus fence box; `S_far` = `S_max` + 32. | 0.5 |
| L1 SEAM | The amendments (§5.5); `hexcanvas` in the enum on both sides; the registry row and schema; `build_canvas` and its goldens; the position-only orientation rule; the ring reader rule; the D6 and container-invariance tests; the conformance entries (§5.4); the stamp and arch kind; a check that the arena and the cells already have a cap outcome for a refused root (else one is designed here). | `make gates`; the R5 pins; the registry census; review. | 2 |
| L2 SERVE | The model (HexConv, line features, LineConv, MaskedBN, heads); the canvas wire; the tier planner and collate checks; CUDA graphs per (tier, batch); bf16; `LeafKey`'s representation word; served-sims exactness for `hexcanvas`; the bench. | The protected served-sims and 1-in-1 collate pins pass for `hexcanvas`; the R3 and R4 pins; the bench against its pre-registered ceiling (§4.4). | 2 |
| L2b TRAIN | The canvas sampler; bucketed microbatches and loss shares; `run_declared_train_step`; the save-time BN-statistics recompute; caps in cells; the lineage weight column read; a content-deduplicated lab-corpus ring set with lineage tags and the occupancy join, built by a tool. | The trainer-seam conformance, the finite-gradient guard and the resume round trip for `hexcanvas`; a 1k-step smoke on the corpus with the witnesses read; review. | 1.5 |
| L3 SCREEN | PRESTATED (rule, controls, witness floors, power from §6.2, the lineage weights); 5 arms read through EMA; panels on rung 16 and S under DEPLOY-2's cap. | §6.2's rule, read as pre-stated. | 1 (+ ≈ 70–120 box-h) |
| L4 RUN | On R391(d)'s trigger and L3's PROCEED: the mint (the dense optimiser, run12's other settings), the preflight, the launch, the deadline converted to steps. | §6.3's fixed-window line, futility halt and run12's halts. | 1 (+ the run) |
| L5 GROW | 10×96 → 10×128 → 15×128, function-preserving, on a pre-stated plateau test. | An equal-steps control fine-tune (LAW-19). | 1 |

**L3's price, from DISTILL-2's measured walls** (4×256 at 0.88 s/step; an S cell ≈ 3.5 h under the cap beside run11):
- the GNN arms take 64k × 0.88 s ≈ 15.6 h each;
- the dense arms take ≈ 5–15 h each, a rough figure scaled from FLOPs, the unit §4.3 rejects; L2b measures it;
- 4 arms × 4 saves × 2 rulers + 2 known-bad cells = 34 cells of 576 games, at 1.5–3.5 h each, 2–3 in flight;
- on one GPU, that is ≈ 70–120 h of box A.

**About 8 sessions to a self-play start, plus reviews.**

## 9. Questions for the architect

1. **The read R390(a) allows.** A binary screen (L3) and then a plateau-break run (L4) never yields a controlled
   capacity verdict. Is that the read the architect wants? The two controlled alternatives are both priced:
   - a converged supervised pair before any run, each arm run until its rung-16 panel is flat (about 4 passes):
     ≈ 30 h per GNN arm, ≈ 150–250 box-h in all;
   - an equal-recipe GNN run beside L4: twice the box for the run's length.
2. **The screen's minimal effect.** −0.34, twice the line, with smaller deficits left to L4. Accept?
3. **Margin.**
   - The fence canvas scores every legal cell (margin 8, Bubble's choice). A margin of 5 with the `far` logit for the
     rest shortens the box side from span + 17 to span + 11.
   - At the median, rounding to a multiple of 8 eats the whole saving (31 and 25 both round to 32); the saving is in
     the tail.
   - Adopt it only if L2 breaches its ceiling and L0 shows under 0.5 % of explicit visit mass beyond 5?
4. **History.** Markov first, with the rivals' one-turn history carded (§3) because it needs a key word, a ring field
   and exposure before it can be read. Accept, or require history in the first row?
5. **Lineage weights.** run11 and its fresh line at 1, run7–run8 at 0.5, run6 at 0.25? A schedule must be pre-stated
   before L3.
6. **Trigger.** May L0 (records only) run before run12's 84k read, so that the shape mix is known if the trigger
   fires?
7. **The cache precondition.** R3 extends R370(c)'s precondition, "a replay inside the served path's own batch-size
   spread", to the spread across canvas tiers, bounded at 5e-3 in policy probability. That extends a ruling: is it
   accepted?
