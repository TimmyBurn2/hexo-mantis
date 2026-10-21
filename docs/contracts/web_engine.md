# Contract: web engine API (#13)

- version: v1
- owner: crate mantis-player (`batch.rs`, the batch both net backends read); `tools/export_web.py` (the manifest and its
  witness rows)
- status: v1 — the batch, its outputs, the export's files, the manifest and the witness lines land with DEPLOY-WEB L1
  (`docs/design/DEPLOY_WEB_2026-10-20.md` §3.1, §4, §5). The step machine (L2) and the typed API, the Worker messages
  and the arena adapter (L3) extend this doc, each bumping its version in the commit that changes it.

## Summary

The browser bot is one engine, `mantis-player`, with two net backends: `RustForward` on one CPU thread and ORT-web on
WebGPU (R398(b)). Both read ONE batch, built in the engine from the served path's own leaf build, fuse and pack
(`build_leaf_graph`, `GraphWire::from_axis_graphs`, `pack_wire`): every array beyond `x` is int32, so no index or mask
is derived inside the net's graph. The export writes, from a stamp, the ONNX graph in fp32 and fp16 over that batch,
the Rust forward's weights, a device witness and a manifest that binds them to the stamp and to every witness row.

## The batch (`mantis_player::batch::WebBatch`; `_engine.web_batch`)

G graphs block-diagonally, rows global: N nodes, E edges, L legal rows. All little-endian.

| array | dtype | shape | meaning |
|---|---|---|---|
| `x` | f32 | [N, node_feat_dim] | node features, as packed (11 at the graph rows) |
| `src`, `dst` | i32 | [E] | each edge's global node rows |
| `code` | i32 | [E] | each edge's row in `edge_vocabulary(win_length)` (91 rows at win length 6) |
| `legal` | i32 | [L] | the legal cells' global node rows, strictly ascending |
| `node_graph`, `legal_graph` | i32 | [N], [L] | the graph of each node and of each legal row |
| `divisor` | i32 | [N] | 1 on a real node (a stone or legal), else max(in-degree, 1) |
| `real` | i32 | [N] | 1 on a real node, else 0 (the value head's max) |
| `mean_mask` | i32 | [N] | 1 on a stone, or on every node of a graph with no stones (the value head's mean) |
| `mean_den` | i32 | [G] | each graph's `mean_mask` count; it also carries G |

A graph's rows are laid out `[stones | legal | dummy]`, as the builder lays them; a stone is a row below the graph's
`n_stones`. The batch is refused (`BatchError`) on no positions, a row without graph geometry (`hexcanvas` by name), a
leaf's seam guard, any pack refusal (checks 4–16 of `graph_wire.md`, check 14 as a coded row outside the
vocabulary), or a count past int32.

## The outputs

| output | dtype | shape | meaning |
|---|---|---|---|
| `logits` | f32 | [L] | the policy head's logit per legal row |
| `probs` | f32 | [L] | the per-graph softmax over its legal rows |
| `value` | f32 | [G] | the 65-bin head's expectation over linspace(−1, 1, 65), clamped to [−1, 1] |

`probs` and `value` are what `served_outputs` returns; `logits` is the witness's.

## The export (`tools/export_web.py`)

Every file is named `<stem>.<kind>`, `<stem>` the stamp's own `{run_id}_{step:08d}_{sha8}` (LAW-12's reading, accepted by
R399(h)), and listed in the manifest with its sha256 and bytes.

| file | contents |
|---|---|
| `<stem>.fp32.onnx` | torch.onnx's dynamo export of the served forward over the batch, opset 18, N, E, L and G dynamic; every scatter-sum is ScatterElements(add), the maxes ScatterElements(max), every gather a Gather; each int32 input reaches its index readers as int32 (`narrow_index_casts`); a ScatterND, a GatherND or an int32 input still cast to int64 refuses the export |
| `<stem>.fp16.onnx` | the same graph with every float initializer fp16 holds finite stored fp16 and cast to fp32 where read (a constant like finfo(f32).min stays fp32); compute stays fp32 |
| `<stem>.weights.bin` | the Rust forward's tensors, raw f32, linear weights transposed (in × out), in the manifest's tensor table's order |
| `<stem>.witness.json` | the device witness: the five openings and eleven goldens spread over stone count, with the served `logits`, `probs` per position and `value` |
| `<stem>.manifest.json` | the manifest below |

## The manifest (`mantis-web-manifest-v1`)

| section | fields |
|---|---|
| `stamp` | `file`, `sha256` of its bytes, `run_id`, `step`, `lineage`, `commit_sha`, `arch_kind`, `encoding`, `weights` (`ema` or `learner`), `net_param_hash` (recomputed; the stamp carries none), `arch` (its widths) |
| `exporter` | `commit`, `dirty`, `torch`, `onnx`, `onnxscript`, `onnxruntime`, `opset`, `export_s` |
| `encoding` | `name`, `registry_sha` (the compiled registry's), `win_length`, `graph_radius`, `trunk_size`, `empty_edges`, `node_feat_dim`, `edge_vocab_rows` |
| `deploy.resolved` | from the stamp's own config through the run's resolvers: `search_kind`, `c_puct`, `fpu_reduction`, `quiescence_enabled`, `quiescence_blend_2`, `leaf_batch_size`, `tactics` (the audit's `mode` included) |
| `inputs`, `outputs` | the names above, in the graph's order |
| `files` | per file: `name`, `sha256`, `bytes`; per ONNX file `dtype`, `ops` (its op census) and `passes_its_line` (every row of that file holds its line); for the weights `tensors` (`name`, `rows`, `cols`, `offset` in floats) |
| `witness` | `lines`; `goldens` (path, sha256, positions, openings); `rows`, one per (backend, file, threads, B): max \|Δ\| `logits`, `probs`, `value`, `argmax_agreed` / `argmax_compared`, `line`, `pass` |

## The detector

A closed table; a value it does not carry is refused by name (`ExportRefused`, its `field`), before anything is written.

| field | carried | refused |
|---|---|---|
| `arch_kind` | `GnnArchV2`, `GnnArchV2SoftPolicy`, `GnnArchV2SoftPolicyFuture` | `GnnArch` (another readout); any other |
| `encoding` | `gnn_axis_r8`, `gnn_axis_r8_pruned` | `hexcanvas_r8` (no model), `gnn_axis_v1` (no goldens at radius 6); any other |
| `deploy.search.kind` | `puct` | `gumbel` (the player carries PUCT only); any other |

The export also refuses (`field`): `registry`, a stamp whose encoding the import-time registry handshake did not check;
`batch`, an engine batch whose `x` is not the served batch's; `re-expression`, one not bit-identical to `served_outputs`
on every golden batch; `graph`, a written graph with a ScatterND, a GatherND or an int32 input cast to int64 (the file
is removed); `stamp` and `config`, a stamp with no arch or config mapping. A venv synced with `--extra teacher
--no-group web-export` has no ONNX toolchain and runs no export.

## The witness lines

The reference is the deploy path's served output: `LocalInferenceEngine` on the CPU, eager, fp32, at its
`served_outputs` call, on the 512 golden positions (`tests/fixtures/graph_build/positions_r8.txt`) at B 1, 8 and 16 and
on the five openings as one batch. A graph's argmax is compared where the reference's top-2 logit gap exceeds the line's.

| line | \|Δ logit\| | \|Δ prob\| | \|Δ value\| | argmax gap | holds for |
|---|---|---|---|---|---|
| fp32 | ≤ 1e-3 | ≤ 1e-4 | ≤ 1e-4 | > 2e-3 | every shipped fp32 backend; a miss at export is a HALT |
| fp16 | reported | ≤ 2e-3 | ≤ 2e-3 | > 2e-2 | the fp16 file, WebGPU only (R399(a)); it ships only after its pre-stated read on the development net |

## Who asserts what where

| fact | asserted where | pinning test |
|---|---|---|
| the batch's derived arrays agree with the offsets they came from; an empty board's mean is every node; the pruned row keeps the nodes; a canvas row and an empty request are refused | `crates/mantis-player/src/batch.rs` | `batch::tests::every_derived_array_agrees_with_the_offsets_it_came_from`, `an_empty_board_takes_the_mean_over_every_node`, `the_pruned_row_builds_fewer_edges_on_the_same_nodes`, `a_canvas_row_and_an_empty_request_are_refused_by_name` |
| every carried kind and row exports, re-expresses bit for bit, and holds the fp32 lines at 1 and default threads | `tools/export_web.py` | `tests/tools/test_export_web.py::test_every_carried_kind_and_row_exports_and_holds_the_fp32_lines_at_one_and_default_threads` (gate 20) |
| no written graph holds a ScatterND, a GatherND or an int32 input cast to int64; an unnarrowed graph is refused and removed (planted break) | `export_web.narrow_index_casts`, `graph_refusal` | `test_no_written_graph_holds_a_scatternd_a_gathernd_or_an_int32_input_cast_to_int64`, `test_an_unnarrowed_graph_is_refused_and_leaves_no_file`, `test_the_graph_refusal_names_each_widening` |
| the fp16 file stores only what fp16 holds finite | `export_web.fp16_storage` | `test_the_fp16_file_stores_only_what_fp16_holds_finite` |
| the manifest binds the stamp, the files and the resolved block | `export_web.export` | `test_the_manifest_binds_the_stamp_the_files_and_the_resolved_block` |
| the witness reds a perturbed graph on the line each break moves, in both lines: logits (bias shift), probs (weight scale), value (one bin) (planted breaks) | `export_web.witness`, `passes` | `test_a_perturbed_written_graph_misses_the_line_it_moves` |
| a drifted re-expression, a foreign batch and an unchecked registry are refused before anything is written (planted breaks) | `export_web.export`, `witness_cases` | `test_a_re_expression_that_is_not_bit_identical_is_refused`, `test_a_foreign_batch_and_an_unchecked_registry_are_refused` |
| the detector refuses by name | `export_web.detect` | `test_the_detector_refuses_what_it_does_not_carry_by_name`, `test_a_gumbel_deploy_block_is_refused_before_anything_is_written` |

## Pinning tests

- `crates/mantis-player/src/batch.rs` (unit tests, gate 2a)
- `tests/tools/test_export_web.py` (gate 20, and the default tier)
