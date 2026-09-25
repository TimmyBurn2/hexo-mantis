# PERF-ADA online research — primary sources for a 4080 SUPER + 9950X serving/training host (2026-09-24)
References re-verified against dev fc37f3f2 (post-SLIM merge) 2026-09-25; measurements unchanged (taken on 51eddda5).

One research agent, online, against primary sources only (NVIDIA headers/docs, PyTorch source at tag
`v2.11.0`, PyG source, LLVM/rustc source and release notes, AMD first-party papers, the first-party READMEs
of the profilers, the AlphaZero preprint, KataGo/lc0/Mctx repos). No repo file changed besides this one;
nothing in it is a decision. Written from worktree `perf-ada` at HEAD `9839becd`. The measured context
(the 4080S box profile and the 1M×128 microbench) is taken as given from the dispatch and from
`PERF_ADA_PLAN_2026-09-24.md` §0.

**Tags.** SOURCED = the cited primary text or code says it. INFERRED = my reasoning from sourced facts plus
the measured numbers; it carries a falsifier. UNKNOWN = the sources do not settle it.
Line numbers are for the fetched file at the stated tag/version (`v2.11.0` for every PyTorch path).

---

## Answers first

**Q1 — bf16 atomics (HIGH).** The CUDA docs say `atomicAdd` on `__nv_bfloat16`/`__nv_bfloat162` is
*available* from cc 8.x and *native* only from cc 9.x; older devices "use emulation path". The header
shipped with the cu128 runtime proves it: below `NV_PROVIDES_SM_90` both overloads compile to a
32-/16-bit `atomicCAS` retry loop, from sm_90 up to `atom.add.noftz.bf16{,x2}`. PyTorch's `index_add_` on
CUDA always goes `ReduceAdd → fastAtomicAdd(..., true) → fastSpecializedAtomicAdd<BFloat16>`, which on
`__CUDA_ARCH__ >= 800` issues `atomicAdd(__nv_bfloat162*)` with the neighbour lane padded with zero. So on
sm_89 (and the wheel ships no sm_89 SASS: it runs the sm_86 cubin) every bf16 element add is a CAS loop; on
sm_120 (a 5080; the cu128 wheel builds sm_120) it is one native bf16x2 atomic. That explains the 4.5× gap:
fp32, native on both, is near-equal (2.43 vs 2.30 ms), so the bf16 gap is the emulation, not the memory
system. The data-dependence is a property of CAS: an add of +0 rewrites the same bits, so a racing CAS still
succeeds and nobody retries. ReLU-sparse (trained) messages are mostly zeros, so they retry less than dense
random-weight messages (INFERRED from the loop's code).

**Q2 — scatter-sum alternatives (HIGH on mechanics, MED on cost).** (a) fp32 buffer + `index_add_`: native
fp32 `RED`, 2.6× faster in the microbench, more accurate, **not** deterministic. (b)
`index_put_(..., accumulate=True)` on CUDA *always* takes a sort-based kernel: a cub radix sort of the
indices, then a per-duplicate-run sum in `opmath_t` (fp32 for bf16) and one rounding. It is deterministic
with fp32 accumulation in bf16 I/O, needs no custom code, and is unmeasured here. `use_deterministic_algorithms(True)`
reroutes `index_add_`/`scatter_add_` into exactly this path. (c) `torch.segment_reduce` on dst-sorted data: one
thread per (segment, channel) looping sequentially. It is deterministic, but it accumulates in the
*storage* dtype (bf16 input means bf16 accumulation). (d) CSR SpMM through `torch.sparse`: PyTorch hard-codes
`CUSPARSE_SPMM_CSR_ALG2`, which cuSPARSE documents as *not* bit-reproducible. (e) PyG sums with plain
`scatter_add_`, the same atomic class. Its fused `spmm` path exists for `GINConv` but not for `GINEConv`,
because the `relu(x_j + e)` message is nonlinear. (f) Inductor: `aten::index_add` is on the fallback list
(opaque aten call). bf16 `index_put(accumulate)`/`scatter_add` fall back to aten on cc < 9.0 by an explicit
rule. **fp32** accumulate lowers to a Triton `atomic_add` scatter whose value loader can inline the
gather+relu producer. Any `amax` scatter falls back. Deterministic mode forces fallback. The only
deterministic options are (b), (c), a custom dst-sorted segment kernel, and cuSPARSE `CSR_ALG3`, which
PyTorch does not expose.

**Q3 — overlap (MED).** The primary docs give the pieces but no recipe: pinned + `non_blocking` copies,
events, stream priorities (hints only, no preemption, no effect on copies). For CUDA graphs, the plain-CUDA
docs say "dynamic shapes are prohibited". CUDAGraph Trees re-record per distinct shape and recommend padding
to a few fixed shapes. The PyTorch docs call `torch.compile(dynamic=True)` "not recommended", because it can
regress performance, and point to `mark_dynamic`. On these numbers the launch overhead CUDA graphs remove is
small next to the GPU time per batch (INFERRED). F-47's un-overlapped CPU/GPU chain is the bigger lever.

**Q4 — Ada specifics (MED).** AD103 has 65 536 KB of L2 (whitepaper App. B). The 4080 SUPER is the full
80-SM AD103. The node-feature table of one batch (56k×128 bf16 ≈ 14 MB) already fits in L2, so L2
persistence controls (`cudaAccessPolicyWindow`, not exposed by PyTorch) have little to add (INFERRED). GeForce
Ada runs bf16/fp16 tensor math with fp32 accumulate at *half* the fp16-accumulate rate (97.5 vs 194.9 dense
TFLOPS on the 4080). GEMMs are 10 % of the profile, so this does not matter here. What PyTorch 2.11 leaves on
the table on Ada is specifically the bf16 atomics: both eager (`fastAtomicAdd`) and Inductor (the cc < 9.0
fallback rule) route bf16 scatter into the CAS loop or away from Triton.

**Q5 — Zen 5 CPU (MED).** `znver5` arrived with LLVM 19, which Rust took in 1.82.0. The pinned 1.97.1 has
it. In LLVM 19 `znver5` = `znver4` features (AVX-512 F/CD/DQ/BW/VL/VBMI/VBMI2/IFMA/VNNI/BITALG/VPOPCNTDQ, BF16,
GFNI, VAES, VPCLMULQDQ) + MOVDIRI/MOVDIR64B/VP2INTERSECT/PREFETCHI/AVX-VNNI. It is scheduled with the Zen 4
model and has no prefer-256-bit tuning. AMD documents a full 512-bit FP/vector datapath and six FP pipes for
the Zen 5 core, which the desktop part shares (INFERRED for the 9950X from the EPYC paper). R2/LAW-13 keep any
of this env-only. PyTorch documents intra-op = one pool per process, sized to the core count by default, and
warns about oversubscription when the application runs its own thread pool. `set_num_threads` must be called
before any torch work runs. Pinned host memory goes through a caching host allocator: blocks are rounded up to
a power of two, never split, and reused only after their recorded events complete. So `pin_memory()` per batch
costs a memcpy, not a `cudaHostAlloc`, after warm-up.

**Q6 — training + serving on one GPU (MED).** NVIDIA's MPS docs say that without MPS, work from different
CUDA contexts "cannot execute concurrently" (time-sliced). MPS needs its control daemon, a same-UID server
(root only for multi-user), a pipe directory, and it *recommends* EXCLUSIVE_PROCESS compute mode (setting that
needs admin). Whether the container ships `nvidia-cuda-mps-control` and allows it is UNKNOWN. One process with
two streams sidesteps it all. Stream priorities are documented as scheduling hints for *pending* kernels
only, with no preemption and no effect on memcpys.

**Q7 — other primary-source levers (MED).** GINE's aggregation can't be written as an SpMM, so the
library-grade fused paths (PyG `spmm`, cuSPARSE) don't apply. The first-party route to a fused, deterministic
kernel inside `torch.compile` is `torch.library.triton_op` + `wrap_triton`. The graph builder makes the
edge features a small categorical code: one-hot axis × signed distance × source-player, plus the zero dummy
vector. That lets such a kernel read a table instead of an E×128 edge tensor, and lets the wire format drop
to a u8 per edge (INFERRED from the builder source).

---

## Q1 — bf16 atomics: hardware, CUDA header, PyTorch path

### What the CUDA documentation says (SOURCED)

- CUDA Programming Guide, appendix 5.4 (C++ Language Extensions), Atomic Functions → `atomicAdd()`
  (<https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html>):
  "`__nv_bfloat16`, `__nv_bfloat162` on devices of compute capability 8.x and higher"; "float2, float4 on
  devices of compute capability 9.x and higher, and only supported for global memory addresses"; vector
  atomicity "is guaranteed separately for each of the components".
- CUDA Math API v13.4, Bfloat162 Arithmetic Functions
  (<https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT162__ARITHMETIC.html>),
  `atomicAdd(__nv_bfloat162*, const __nv_bfloat162)`: "This operation is natively supported by devices of
  compute capability 9.x and higher, older devices use emulation path."
- So 8.x (incl. Ada sm_89) = supported **by emulation**, 9.x+ (Hopper sm_90, Blackwell sm_100/sm_120) =
  native.

### What the header actually compiles to (SOURCED)

`include/cuda_bf16.hpp` from the `nvidia-cuda-runtime-cu12` **12.8.90** wheel (the runtime a
`torch 2.11.0+cu128` install pulls; the copy Triton bundles is identical at these lines):

- Lines 3802–3821, `atomicAdd(__nv_bfloat162 *const address, const __nv_bfloat162 val)`:
  `NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, atom.add.noftz.bf16x2 …, <else>)`. The else branch is a loop:
  `assumed = old; new_val = __hadd2(val, assumed); old = atomicCAS(address_as_uint, assumed, new_val);`
  `while (assumed != old)`.
- Lines 3823–3845, `atomicAdd(__nv_bfloat16 *const address, …)`: same shape. Native `atom.add.noftz.bf16`
  on SM_90+, else a 16-bit `atomicCAS` loop.
- Contrast, `cuda_fp16.hpp` line 3426–3433: `atomicAdd(__half2*)` is native (`atom.add.noftz.f16x2`) from
  `NV_PROVIDES_SM_60`. fp16 is native on Ada and bf16 is not. See the F-11/LAW-06 interaction below.

### What PyTorch 2.11 does for `index_add_` (SOURCED)

- `aten/src/ATen/native/cuda/Indexing.cu` L516–525: `class ReduceAdd` calls
  `fastAtomicAdd(self_data_start, index, numel, *src_data, true)` on CUDA (the `opportunistic_` variant is
  gfx942/gfx950 only). L1130–1152 `index_add_cuda_impl`: if `globalContext().deterministicAlgorithms()`, it
  instead calls `result.index_put_(indices, source * alpha, true)` (L1143–1151). Otherwise it launches
  `indexFuncSmallIndex` (numIndex ≤ 16) or `indexFuncLargeIndex` (L1221–1250), dispatched over
  `AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(Bool, Half, BFloat16, ComplexHalf)` and `AT_DISPATCH_INDEX_TYPES`
  (int32 and int64 indices both valid).
- `aten/src/ATen/native/cuda/KernelUtils.cuh` L155–196, `fastSpecializedAtomicAdd<BFloat16>`: under
  `__CUDA_ARCH__ < 800` it uses `gpuAtomicAddNoReturn` (PyTorch's own CAS). Otherwise it builds a
  `__nv_bfloat162` with the value in its lane and `NATIVE_ZERO_BF16` in the other, and calls `ATOMICADD`
  (= `atomicAdd` on CUDA, L77) on the aligned pair (L171–185). L213–223: `fastAtomicAdd(..., true)` always
  takes that specialised path.
- `aten/src/ATen/cuda/Atomic.cuh` L222–231 `gpuAtomicAdd(at::BFloat16*)`: `USE_ROCM || __CUDA_ARCH__ < 800`
  means `AtomicFPOp` CAS (L38–58), else `atomicAdd(__nv_bfloat16*)`. **L436–441 `gpuAtomicMax(at::BFloat16*)`
  is `AtomicFPOp` CAS on every architecture.** That is the path of the model's bf16
  `scatter_reduce_(…, "amax")` (`ScatterGatherKernel.cu` L63–68 → `gpuAtomicMax`).
- Wheel architectures, `.ci/manywheel/build_cuda.sh` L107–111: base `7.5;8.0;8.6;9.0;10.0`, with `12.0`
  added for CUDA 12.8. There is **no 8.9 cubin**, so a 4080S runs the sm_86 SASS (`__CUDA_ARCH__` = 860:
  the emulated branch). A 5080 runs the sm_120 SASS: the native branch.

### Result per architecture (SOURCED chain)

| arch | `index_add_` bf16 element add | `scatter_reduce_` bf16 amax |
|---|---|---|
| sm_89 (4080S; runs sm_86 SASS) | `atomicAdd(bf16x2)` → **32-bit CAS loop** (cuda_bf16.hpp L3811–3819) | `AtomicFPOp` CAS |
| sm_90 (H100) | native `atom.add.noftz.bf16x2` | `AtomicFPOp` CAS |
| sm_100 / sm_120 (Blackwell) | native `atom.add.noftz.bf16x2` | `AtomicFPOp` CAS |

### Does this explain 6.25 ms (4080S) vs 1.40 ms (5080)? (INFERRED, HIGH)

Yes. fp32 atomics are native on both and read 2.43 vs 2.30 ms, so the two memory systems deliver roughly
the same atomic throughput. The bf16 gap (4.5×) is therefore not bandwidth; it is the CAS emulation. The
5080 bf16 beating its own fp32 (1.40 vs 2.30) fits the native bf16x2 op moving half the bytes.

Two structural contention sources on top of the graph's own fan-in (INFERRED from the code above):
1. **Built-in 2-way pairing.** PyTorch pads each element into a bf16x2 word. The threads for channels `c` and
   `c+1` of one edge row therefore CAS the *same* 32-bit word, and every element add sees at least one
   potential collision on sm_89.
2. **The dummy node.** The builder adds bidirectional edges from a dummy node to every real node
   (`crates/mantis-graph/src/lib.rs` L740–748). With ≈ 875 nodes/graph (56k/64), each graph's dummy row
   receives ≈ 875 messages, so 64 rows take ≈ 56k of the 1M adds under heavy same-address contention.
   Contention is where a CAS loop is at its worst. The same shape recurs in the readout: `gnn_v2.py` L52/L55
   (`scatter_reduce_` amax into 64 rows over 56k nodes) and L58 (`counts.index_add_` into 64 slots).

### Why is it data-dependent? (INFERRED from the header loop, MED-HIGH)

The loop retries only when `atomicCAS` returns a value different from `assumed`, that is, when another
thread *changed* the word in between. A thread adding `+0` writes back the same bits, so a racing thread
whose `assumed` predates that write still matches and succeeds (a benign ABA). Trained GINE messages are
`relu(...)` outputs, mostly exact zeros. Random-weight messages are dense. Fewer effective writes mean fewer
retries. Falsifier: the microbench with `msg` masked to 0 %, 50 % and 90 % zeros on the 4080S. Time should
fall with the zero fraction; the same sweep on the 5080 (native) should be flat.

**Numerics side note (SOURCED + INFERRED).** The current path accumulates in bf16 with one rounding per add,
in a nondeterministic order. The dummy row sums ≈ 875 terms that way. The plan doc's measured max error
(0.375 vs fp32) fits. An aside to verify, not a finding: `gnn_v2.py` L79–80 accumulates `in_degree` in
`x.dtype`. If that were ever bf16, `+1` accumulation stalls at 256 (8-bit significand). With fp32 input
features, as the collate emits, it is exact.

---

## Q2 — scatter-sum alternatives on sm_89 in PyTorch 2.11

| option | mechanism (SOURCED) | accumulate precision | deterministic? | cost on this host |
|---|---|---|---|---|
| bf16 `index_add_` (today) | `fastAtomicAdd` → CAS loop on sm_89 (Q1) | bf16, one rounding per add | **no** (order-dependent rounding under contention) | measured 6.25 ms random / ≈ 4.2 ms trained per call |
| fp32 buffer + `index_add_` | native fp32 atomics (`gpuAtomicAddNoReturn` → `atomicAdd(float*)`) | fp32 | **no** | measured 2.43 ms incl. cast |
| `index_put_((dst,), msg, accumulate=True)` | `TensorAdvancedIndexing.cpp` L988–1003: on CUDA, `accumulate` **always** routes to `index_put_with_sort_stub`. `Indexing.cu` L661–700: `cub::radix_sort_pairs` on `nbits = log2(max index)`, then `indexing_backward_kernel*` summing each duplicate run in `opmath_t` (L147–252; fp32 for bf16) and writing `self + sum` once | fp32 per destination, then one bf16 rounding | **yes** (stable sort; per-run sequential sum) | **UNMEASURED.** A 1M-key radix sort plus one pass; the sort repeats in each of the 4 layers on the same `dst` |
| deterministic mode | `index_add_cuda_impl` L1143–1151 and `scatter_add` (`TensorAdvancedIndexing.cpp` L2332–2341) reroute to the same `index_put_` sort path | same as above | yes | same as above, but global: it forces Inductor fallbacks (below) and makes some other ops raise |
| `torch.segment_reduce` (beta) on dst-sorted rows | `SegmentReduce.cu` L92–130: for 2-D data, one thread per (segment, channel), loop `j = offset_start … offset_end`. `initial_value` is `scalar_t` | **storage dtype** (bf16 in = bf16 sequential accumulate) | yes | measured 2.11 ms (fp32, pre-sorted). Needs edges sorted by `dst` and the message tensor in that order |
| `torch.sparse` CSR SpMM (incidence N×E times msg) | `SparseBlasImpl.cpp` L564: `auto algorithm = CUSPARSE_SPMM_CSR_ALG2`, compute type `opmath_t` (L579) | fp32 | **no**: cuSPARSE docs list CSR_ALG2 as "May produce slightly different results during different runs"; only `COO_ALG2`, `CSR_ALG3`, `BSR_ALG1` are "deterministic (bit-wise)" (<https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespmm>) | UNMEASURED. `CSR_ALG3` is not reachable from PyTorch |
| PyG `torch_geometric.utils.scatter(reduce='sum')` | `_scatter.py`: `src.new_zeros(size).scatter_add_(dim, index, src)` (the sum branch); `torch_scatter` is used only for min/max/mul with grad on CUDA | as `scatter_add_` | no (same atomic class) | same class as `index_add_` |
| PyG fused `message_and_aggregate` | `gin_conv.py`: `GINConv.message_and_aggregate` → `spmm(adj_t, x, reduce)` (L95–98); **`GINEConv` defines only `message(x_j, edge_attr)`** (L195) | n/a | n/a | not applicable: GINE's `relu(x_j + e)` is nonlinear, so no SpMM form exists |
| dst-sorted fused segment kernel (custom Triton) | not in any library. See Q7 | fp32 in registers | yes (fixed per-row order) | INFERRED to be the floor: no `msg` round-trip, no atomics |

**How Inductor lowers these (SOURCED, `torch/_inductor`):**
- `lowering.py` L104–109: `FALLBACK_ALLOW_LIST` contains `"aten::index_add"`, so a compiled `index_add_` is
  an opaque aten call and the eager CAS path. The profile's `index_add_` line under `torch.compile` fits.
- `lowering.py` L4066–4068: in deterministic mode, `index_put_` falls back. L4078–4084: `accumulate and
  needs_fallback_due_to_atomic_add_limitations(dtype)` also falls back. `utils.py` L3215–3221: that function
  returns `True` for bf16 when `torch.cuda.get_device_capability() < (9, 0)`. **On Ada every bf16 scatter-add
  is an aten fallback by rule.**
- Otherwise (fp32) `index_put` accumulate becomes `ir.Scatter(..., inner_fn=values.make_loader(),
  scatter_mode="atomic_add")` (L4125–4133): a Triton kernel with atomics. The value loader can inline an
  un-realized pointwise producer, which would fuse the `index_select + e → relu` into the scatter kernel and
  drop the materialized E×128 `msg` (INFERRED; confirm in the generated code with `TORCH_LOGS=output_code`).
- `utils.py` L3224–3257 `use_scatter_fallback`: fallback for any reduction other than sum/add (so `amax`
  scatters are aten), for bf16 on cc < 9.0, and whenever deterministic algorithms are on.
- `config.py` L992–1005: `partitioned_scatter_enabled` (env `TORCHINDUCTOR_PARTITIONED_SCATTER_ENABLED`,
  default off), "partitioned scatter optimization for atomic add kernels … at cost of memory usage". This is
  aimed at exactly the hot-row contention the dummy node creates. UNKNOWN whether it triggers for this shape.
- `config.py` L852–854: `deterministic` (env `TORCHINDUCTOR_DETERMINISTIC`) only skips on-device benchmarking
  that affects numerics. It does not make scatters deterministic.

**Determinism list (SOURCED, `torch.use_deterministic_algorithms`, docs 2.11,
<https://docs.pytorch.org/docs/2.11/generated/torch.use_deterministic_algorithms.html>):** "act
deterministically when mode=True" includes `torch.index_add()` on CUDA, `scatter_add_()` on CUDA,
`scatter_reduce()` sum/mean on CUDA and `index_put()` accumulate on CPU. `scatter_reduce()` prod on CUDA
throws. `index_reduce` is not listed. Its CUDA impl calls `alertNotDeterministic("index_reduce_cuda")`
(`Indexing.cu` L1291). `amax` via atomic max is order-independent in value (max is exact), so it is
deterministic by construction, but slow (CAS) (INFERRED).

---

## Q3 — overlapping CPU and GPU in the serving loop

- **Pinned + async (SOURCED).** CUDA semantics notes, "Use pinned memory buffers"
  (<https://docs.pytorch.org/docs/2.11/notes/cuda.html>): H2D from page-locked memory is much faster, and
  `pin_memory()` returns a copy in a pinned region. The same page lists the allocator knobs
  `pinned_use_cuda_host_register`, `pinned_num_register_threads`, `pinned_use_background_threads` and
  `pinned_reserve_segment_size_mb`. The tree already stages pinned + `non_blocking` both ways
  (`graph_collate.py` L425–440, `inference_server.py` L149–155). What F-47 found missing is the *schedule*:
  the server never has a second batch collating while the first runs.
- **Streams (SOURCED).** `torch/cuda/streams.py` L29–33: priority "can be positive, 0, or negative. A lower
  number indicates a higher priority", default 0, clamped to the device range. The CUDA Programming Guide
  (stream priorities, <https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html>;
  Runtime API §6.32 Stream Management) says priorities affect only which *pending* compute kernels launch
  first. They do not preempt running work and have no effect on host↔device memcpys.
- **CUDA graphs (SOURCED).** CUDA semantics: "Dynamic shapes are prohibited. The graph assumes every tensor
  in the captured op sequence has the same size and layout in every replay." CUDAGraph Trees
  (`docs/source/user_guide/torch_compiler/torch.compiler_cudagraph_trees.md` L171–187 at v2.11.0): trees
  "re-record CUDAGraph for every unique shape"; "Nvidia uses 64 KB of device memory per kernel launch in
  CUDAGraph, up until CUDA 12.4 and Driver Version 550+"; "we suggest padding input tensors to a few fixed
  tensor shapes"; `torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True` skips dynamic-shape
  functions. The same file lists functions it skips (input mutation, CPU ops, incompatible ops) and describes
  `graph_partition=True` to split CPU ops off. `torch.compile` docs: `mode="reduce-overhead"` "reduces the
  overhead of python with CUDA graphs, useful for small batches … at the cost of more memory usage".
- **`dynamic=True` (SOURCED).** `torch.compiler_dynamic_shapes.md` (v2.11.0) under
  "`torch.compile (dynamic=true)` (Not recommended)": it "forces all sizes and integers to be dynamic …
  may result in performance regressions and ultimately increase compilation time". The recommended tools
  are automatic dynamic (compile static, mark changed dims dynamic on the first recompile) or
  `torch._dynamo.mark_dynamic(t, dim, min=, max=)` before the call.
- **Applied to this host (INFERRED).** A GPU batch here is tens of ms: 55 % of GPU time is ≈ 4 × 4.2 ms of
  `index_add_`, so ≈ 30 ms per batch; F-47 measured ≈ 0.25 ms/graph forward in run6's regime. Kernel-launch
  overhead for a 4-layer GINE is ≈ 100 launches × a few µs, so CUDA graphs buy ≤ a few % here. Padding
  E and N to buckets, with sink rows for padded edges, also costs extra atomic traffic. The larger, sourced
  lever is F-47's overlap: two batches in flight, with the result D2H `non_blocking` into pinned memory and
  an event wait *after* the next collate is issued. Once overlapped, the cycle falls from CPU + GPU to
  max(CPU, GPU). The Rust collate must run inside `py.detach` for any of this to overlap, which is F-46's
  lesson. UNKNOWN: whether the sort-based `index_put_` path (Q2) contains a host sync that would block graph
  capture. Its bounds check (`makeLinearIndex(..., !unsafe)`) is a candidate; check it with
  `torch.cuda.set_sync_debug_mode("error")`.

---

## Q4 — Ada-specific

- **L2 (SOURCED).** Ada whitepaper v2.02 (<https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf>),
  Appendix B: the full AD103 has "80 SMs … and 65,536 KB L2 cache". The RTX 4080 (76 SMs) lists
  716.8 GB/s at 22.4 Gbps. The 4080 SUPER product page lists 10 240 CUDA cores (= 80 SMs × 128, the full
  AD103) but no bandwidth figure (UNKNOWN from first-party; 23 Gbps × 256-bit would be 736 GB/s, INFERRED).
- **L2 persistence (SOURCED + INFERRED).** The Ada tuning guide, "L2 Cache" (under §1.4.2)
  (<https://docs.nvidia.com/cuda/ada-tuning-guide/index.html>) points to the Programming Guide's L2
  persistence controls. PyTorch has no API for them. A user report that set-aside did not work from a
  PyTorch extension is <https://github.com/pytorch/pytorch/issues/101489>. INFERRED low value: the reused
  operand (`x`, 56k×128 bf16 ≈ 14 MB; fp32 agg ≈ 28 MB) already fits in 64 MB. The streaming operand (the
  E×128 `msg`/`e`, ≈ 256 MB bf16) is exactly what persistence would mark streaming, and the better cure is
  not materializing it at all (Q7).
- **Tensor rates (SOURCED, whitepaper App. B, RTX 4080):** BF16 tensor with FP32 accumulate 97.5 TFLOPS
  dense vs FP16-accumulate 194.9; TF32 48.7; FP8 with FP32 accumulate 194.9. GEMMs are 10 % of the profile
  and the edge Linear has K = 5 (memory-bound), so precision knobs on GEMMs are not the lever (INFERRED).
- **Backend knobs (SOURCED, CUDA semantics 2.11):** `torch.backends.cuda.matmul.allow_tf32` defaults
  False, `cudnn.allow_tf32` True, the newer per-backend `torch.backends.fp32_precision` ("ieee"/"tf32"), and
  `allow_bf16_reduced_precision_reduction` is True by default. Under bf16 autocast these touch only ops
  left in fp32.
- **Shared memory / occupancy (SOURCED, Ada tuning guide, "Occupancy" (§1.4.1) and "Shared Memory Capacity" (§1.4.2)):** 48 warps/SM max; up to 99 KB
  shared per block, carveouts 0/8/16/32/64/100 KB. A 128-wide edge-code table in fp32 is
  (codes × 512 B), well inside that (Q7).
- **What PyTorch leaves on the table on Ada (SOURCED):** the cc < 9.0 bf16 routing, in both eager
  (`KernelUtils.cuh` via `cuda_bf16.hpp`'s CAS) and Inductor (`needs_fallback_due_to_atomic_add_limitations`).
  Nothing else Ada-specific surfaced in the sources.

---

## Q5 — Zen 5 / 9950X

- **rustc `znver5` (SOURCED).** Rust `RELEASES.md`: Version 1.82.0 (2024-10-17) carries "Update to LLVM 19"
  (rust-lang/rust#127513). LLVM 19.1.0 `llvm/lib/Target/X86/X86.td` L1548–1557 defines `ZN5Features =
  ZN4Features + [VNNI, MOVDIRI, MOVDIR64B, VP2INTERSECT, PREFETCHI, AVXVNNI]` with `ZN5Tuning = ZN4Tuning`.
  L1529–1545 `ZN4AdditionalFeatures` = AVX512 (F), EVEX512, CDI, DQI, BWI, VLX, VBMI, VBMI2, IFMA, VNNI,
  BITALG, GFNI, BF16, SHSTK, VPOPCNTDQ. L1908: `ProcModel<"znver5", Znver4Model, …>`, the Zen 4 scheduling
  model. The ZN tuning list (L1492–1506) has no `TuningPrefer256Bit`, so LLVM will emit 512-bit vectors
  freely. The pinned 1.97.1 toolchain (LLVM 21 since 1.91) has it. **R2 / LAW-13:** only via `make
  build.native`, env-only, never committed.
- **Zen 5 AVX-512 (SOURCED, AMD first-party, server part):** the "5th Gen AMD EPYC Processor Architecture"
  white paper (<https://docs.amd.com/api/khub/documents/UIqhAbjRhgnzgzzdVU4pUw/content>): "the
  floating-point and vector ALUs have been equipped with a full, 512-bit data path. Data for AVX-512
  operations can load in a single-cycle, and a total of six pipelines keep more floating-point instructions
  in flight"; "implement the full set of AVX-512 instructions used in 4th Gen Intel Xeon processors except
  for FP16 data types"; "BIOS settings can direct the processor to execute two 256-bit vectors in sequential
  clock cycles for AVX-512 instructions." That the desktop Granite Ridge core (9950X) has the same full-width
  datapath is INFERRED. AMD's Zen 5 Software Optimization Guide (pub. 58455) is the owning source but was not
  retrievable (download-only). UNKNOWN for this box: whether its firmware has the 256-bit mode set.
- **What it could buy here (INFERRED, LOW).** The serving CPU half is Python + Rust collate + GIL traffic
  (F-47/F-46): pointer-chasing and memcpy, not SIMD math. Expect a small single-digit % on the Rust collate at
  best. Falsifier: the plan's native-build A/B (one commit, LAW-09 IQR bench).
- **PyTorch CPU threads (SOURCED).** The 2.5 note "CPU threading and TorchScript inference"
  (<https://docs.pytorch.org/docs/2.5/notes/cpu_threading_torchscript_inference.html>; stubbed in 2.11):
  default intra-op threads = number of CPU cores; "avoid oversubscription … in an application that uses a
  large application thread pool … one might find disabling intra-op parallelism as a possible option"; "two
  different application or inter-op threads may use different OpenMP thread pools". `torch/_torch_docs.py`
  L9815–9838: `set_num_threads` "must be called before running eager, JIT or autograd code";
  `set_num_interop_threads` "Can only be called once and before any inter-op parallel work is started".
  INFERRED: with 32 Rust/Python workers on 32 hardware threads, a 16-thread intra-op pool (used by
  `graph_collate.py`'s ≥ 1 MB `pin_memory()` path, L421/L437) oversubscribes. Measure intra-op = 1…4.
- **Pinned allocation cost (SOURCED).** `aten/src/ATen/cuda/PinnedMemoryAllocator.h`:
  `getPinnedMemoryAllocator()` returns `at::getHostAllocator(kCUDA)`, the caching host allocator.
  `aten/src/ATen/core/CachingHostAllocator.h` Note [HostAllocator design] (L156–223): a free list, an event
  queue, allocations served from the free list first; "does not split larger allocations into smaller
  blocks". L299–301: "Round up the allocation to the nearest power of two to improve reuse". A block returns
  to the free list only once every recorded event on it is ready. So after warm-up, per-batch
  `pin_memory()`/`empty(pin_memory=True)` costs a free-list pop plus a **host memcpy**. The memcpy, not the
  pinning, is the per-batch price. It can be removed by having the Rust collate write directly into a
  reused pinned buffer (INFERRED).

---

## Q6 — trainer and server on the same GPU

- **Two processes (SOURCED).** MPS "Architecture" (<https://docs.nvidia.com/deploy/mps/architecture.html>):
  without MPS, "Work launched to the compute engine from work queues belonging to different CUDA contexts
  cannot execute concurrently" (time-sliced).
- **MPS requirements (SOURCED).** "When to Use MPS" (<https://docs.nvidia.com/deploy/mps/when-to-use-mps.html>):
  Linux only, 64-bit apps, "Only one user on a system may have an active MPS server", clients must have the
  server's UID (multi-user only with a root server), and EXCLUSIVE_PROCESS compute mode *recommended*.
  Pipes and sockets live in `CUDA_MPS_PIPE_DIRECTORY` (default `/tmp/nvidia-mps`). The Quick Start is just
  `nvidia-cuda-mps-control -d`, with no privilege statement. INFERRED: a single-UID MPS daemon is not
  documented as needing root, but setting the compute mode is an admin act. UNKNOWN whether the container
  image carries the daemon and whether the host driver's MPS works under the container's namespaces.
  Falsifier: `nvidia-cuda-mps-control -d` with a private pipe dir, then two processes running concurrent
  kernels in an nsys trace.
- **One process, two streams (SOURCED + INFERRED).** A higher-priority serving stream wins only among
  *pending* kernels. Long trainer kernels already running are not preempted and copies are unaffected
  (Q3). In one process both share one caching allocator and one GIL, and F-44's regime annotation governs
  how much trainer time there is to hide.
- **Training-side note (INFERRED from the Q1/Q2 sources):** `index_select`'s backward is an `index_add`, so
  the trainer's gather backward hits the same bf16 CAS path on sm_89. F-44 measured `index_add_` at 22 % of
  a standalone step in run6's regime, and one aggregation fix serves both halves (the plan's R10 note).

---

## Q7 — other primary-source levers for 128-wide GNN inference on Ada

- **Custom fused kernel inside `torch.compile` (SOURCED).** "Using User-Defined Triton Kernels with
  torch.compile" (<https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html>):
  `torch.library.triton_op` + `wrap_triton`; "torch.compile traces into triton_op" (vs opaque
  `custom_op`); supports dynamic shapes and `autograd.Function` (2.3+). No Triton tutorial ships a segment
  reduction (UNKNOWN beyond that).
- **Edge features are a small categorical code (SOURCED, the repo's builder).** `push_attr`
  (`crates/mantis-graph/src/lib.rs` L794–800): `[one_hot(axis∈3), signed_dist, src_player]`, where
  `signed_dist = ±1…±window` (L708) and `src_player = player_feat()` ∈ {+1, −1, empty} (L228–236), plus the
  all-zero dummy vector (L740–748). The number of distinct rows is ≤ 3 · 2·window · 3 + 1, which fits a
  u8 for window ≤ 14.
  INFERRED consequences:
  1. `e = lin(edge_attr)` is a lookup into a (codes × 128) table computed once per forward. A fused kernel
     keeps it in shared memory instead of materializing E×128 (≈ 256 MB bf16) and reading it back.
  2. The wire format can ship `edge_code: u8` + `edge_index: int32` instead of 5 fp32 + 2 int64, a ≈ 4×
     smaller H2D (the measured 5 %) and host memcpy (Q5). The GPU rebuilds the *same* 5-float rows by a table
     gather, so the input to `lin` is bit-identical. `index_select`/`index_add_` accept int32 indices
     (`AT_DISPATCH_INDEX_TYPES`, `Indexing.cu`).
- **The fused, deterministic kernel this points to (INFERRED, the floor).** Edges sorted by `dst` once per
  batch (CSR row pointers). A Triton program per block of destination rows loops the row's in-edges in fixed
  order and computes `relu(x[src] + T[code])` in registers. `x` is L2-resident at 14 MB. It accumulates in
  fp32 and writes the N×128 row once. That removes the atomics, the E×128 `msg` write+read, the separate
  gather+relu kernel (11 %) and the edge-linear output, and it is bitwise deterministic. The dummy row is the
  one long segment: split it or give it its own program. Backward (for the trainer) is a gather-transpose:
  sorting by `src` gives the same kernel shape for `dL/dx`.
- **Not applicable, with reason (SOURCED):** PyG `spmm`/`message_and_aggregate` (GINE has no linear form,
  above); cuSPARSE SpMM through PyTorch (ALG2, non-deterministic); `pyg-lib` segment matmul (typed Linear
  for heterogeneous graphs, a different problem).

---

## Outside the box

Each entry: source, transfer to a 4-layer × 128 GINE over ≈ 15.7k edges/graph (≈ 875 nodes), cheap
falsifier.

1. **A cross-search NN evaluation cache (KataGo, lc0).** SOURCED: KataGo `cpp/configs/gtp_example.cfg`
   L368–378: "KataGo will cache up to (2 ** nnCacheSizePowerOfTwo) many neural net evaluations in case of
   transpositions in the tree", ≈ 1.5 KB/entry. Its self-play config `cpp/configs/training/selfplay1.cfg`
   runs `nnCacheSizePowerOfTwo = 21` (L121) with `nnMaxBatchSize = 128` (L120), `numGameThreads = 128`
   (L84), `numSearchThreads = 1` (L116). lc0 `src/neural/shared_params.cc` L63–83: `NNCacheSize`, default 2 000 000. The
   tree has a per-search transposition table (`crates/mantis-search/src/mcts/mod.rs` L76, cleared L171) but
   no process-wide eval cache was found. Transfer: every hit is a leaf that skips collate + GPU entirely, on
   exactly the chain F-47 says converts 1:1. Caveat: the net is not D6-equivariant (run8@18k symmetry spread
   0.146), so a *canonicalized* key changes outputs, and only exact-position keys are output-neutral.
   Falsifier (0 box-h): log the position hash of every served leaf over one burst and count exact
   duplicates within and across the search trees of one game. Under ≈ 5 % the lever is dead.
2. **Batching across games is already the design (SOURCED):** Mctx's README says its "Search algorithms in
   Mctx are defined for and operate on batches of inputs, in parallel". KataGo self-play: 128 game threads
   feeding one NN server thread. F-47 already falsified "more leaves in flight" at this regime. No new lever
   here.
3. **FP8 GEMMs on Ada.** SOURCED: `ScaledBlas.cpp` L474–475 (v2.11.0): "`torch._scaled_mm` is only
   supported on CUDA devices with compute capability >= 9.0 or 8.9". Whitepaper: FP8 with fp32 accumulate
   194.9 TFLOPS on the 4080. Transfer: GEMMs are 10 % of GPU time, so the ceiling is ≈ 5 %, and it collides
   with LAW-06 (pinned bf16). Not worth a ruling. Falsifier: none needed at 10 %.
4. **CPU inference on Zen 5 AVX-512 BF16 (oneDNN).** SOURCED: LLVM's ZN4/ZN5 features include `BF16` and
   `VNNI`, and AMD states full-width AVX-512. Transfer: backwards for serving. The CPU half is already the
   bound (F-47) and a 1M-edge scatter per batch is GPU work. The one place it could matter is off-run
   single-position evaluation (ladder/analyzer), not self-play. Falsifier: the existing
   `tools/strix_driver.py`/analyzer thread knobs at batch 1.
5. **Split inference and training across devices (AlphaZero).** SOURCED: Silver et al. 2017
   (arXiv:1712.01815) p. 4: "using 5,000 first-generation TPUs to generate self-play games and 64
   second-generation TPUs to train the neural networks". Transfer: removes the time-slicing/stream contention
   of Q6 entirely, but it is a box decision, which **R11 makes the operator's act**. This report states it
   only as the alternative with its cost: a second GPU's rent plus weight shipping at every promotion.
   Falsifier: the trainer's GPU-busy share in-run. If it is small (F-44's regime: asleep 92 %), splitting
   buys little.
6. **Incremental evaluation of sibling leaves.** INFERRED **dead** from the builder: the dummy node connects
   bidirectionally to every real node (L740–748), so after 2 message-passing layers every node's receptive
   field is the whole graph, and a one-stone change touches every embedding by layer 4. No reuse is
   possible without an architecture change.
7. **Graph sparsification / edge pruning.** Source: only the builder itself. INFERRED: dummy edges are
   2 × 875 × 64 ≈ 112k of ≈ 1M (≈ 11 %), axis edges the rest (walk up to `window` per direction, stopping at
   an opposing stone). Any pruning changes the model, so it needs a trained net to evaluate. It is
   STRENGTH_RESEARCH_2's D-2 "edge-cutoff" (ranked #10, low confidence). Falsifier: the regression cell D-2
   already names.
8. **Shrink the wire (INFERRED, exact):** u8 edge codes + int32 indices (Q7). A ≈ 4× cut in H2D bytes and
   in the host pinned memcpy on the CPU-bound chain, with bit-identical model input. Falsifier: bench_server
   A/B, LAW-09.
9. **fp16 instead of bf16 for the aggregation only.** SOURCED: `__half2` atomics are native from sm_60
   (`cuda_fp16.hpp` L3426–3433). Transfer: blocked. LAW-06 pins bf16 on the graph path, F-11 is the fp16
   NaN incident, and the dummy row's ≈ 875-term sum risks fp16's 65 504 ceiling. Listed so nobody
   re-discovers it without the history.

---

## Profiling tooling in an unprivileged container

- **py-spy (SOURCED, README <https://github.com/benfred/py-spy>).** `--native` is supported on Linux
  x86-64 ("For best results, you should compile your Python extension with symbols"). Without root, py-spy
  can profile only a process **it launches** (`py-spy record -- python …`); attaching by PID needs root or
  a relaxed `ptrace_scope`. In containers, `process_vm_readv` usually needs `SYS_PTRACE`. The default
  sampling *pauses* the target; `--nonblocking` avoids that. `--gil` shows only GIL holders and "will miss
  activity in extensions that release the GIL", which matters because the Rust verify/collate runs under
  `py.detach` (F-46's repair). `--idle` includes sleeping threads. INFERRED for Rust frames through PyO3:
  py-spy unwinds the `.so` like any native extension, so the maturin profile must keep symbols (no
  `strip`); `debug = "line-tables-only"` gives file:line. The frame for the Rust function appears under the
  CPython call that entered it.
- **pprof-rs (SOURCED, README <https://github.com/tikv/pprof-rs>).** Uses `setitimer` → `SIGPROF` and a
  signal-handler backtrace, so no perf_event is needed. Its caveats: libgcc's unwinder "is not signal safe"
  (deadlock risk while an exception propagates); frame-pointer mode needs nightly `-Z build-std`. INFERRED
  for use inside the Python process: `ITIMER_PROF` is process-wide, so it samples Python threads too (as
  unsymbolized interpreter frames). Keep it to a Rust-only harness (e.g. a `cargo bench` binary).
- **samply (SOURCED, README <https://github.com/mstange/samply>).** "On Linux, samply uses perf events",
  needing `perf_event_paranoid` ≤ 1 or `CAP_PERFMON` (reported mixed). An unprivileged container cannot
  set either, so it is **not usable** unless the host already allows it. Check `/proc/sys/kernel/perf_event_paranoid`.
- **Nsight Systems (SOURCED, User Guide and Installation Guide, <https://docs.nvidia.com/nsight-systems/UserGuide/index.html>).**
  "To collect thread scheduling data and IP … samples, the Linux operating system's `perf_event_paranoid`
  level must be 2 or less." In a container, perf_event_open needs `--privileged`, `--cap-add=SYS_ADMIN`, or a
  seccomp profile allowing it. `nsys status --environment` reports readiness from inside the container.
  `--sample=none` / `--cpuctxsw=none` switch off the perf-backed collectors. GPU Metrics sampling "Must have
  elevated permissions (see ERR_NVGPUCTRPERM) or be root". INFERRED (not stated verbatim): CUDA API and kernel
  tracing (`-t cuda,nvtx`) is CUPTI-injection based and works with sampling off.
- **torch.profiler / CUPTI (SOURCED, <https://developer.nvidia.com/ERR_NVGPUCTRPERM>).** The admin
  restriction covers GPU *performance counters*: Nsight Compute, nsys GPU metrics, CUPTI profiling/range
  profiler/PM sampling. "CUPTI activity/tracing API" is not restricted. Lifting the restriction needs
  `NVreg_RestrictProfilingToAdminUsers=0` (modprobe, root) or R610+ `/dev/nvidia-caps` access, and for
  containers "access must be enabled on the host". So `torch.profiler` kernel timelines (Kineto's CUPTI
  activity path) work unprivileged, and Nsight Compute kernel counters (e.g. to *prove* the CAS retry count
  on `index_add_`) do not unless the host already allows them. The unprivileged proof of Q1 is instead the
  zero-fraction microbench sweep.

---

## Candidate levers

Effects are INFERRED from the measured numbers above unless marked. Per LAW-09 each is one change, one
commit, one IQR-gated bench with a pre-registered bracket.

| lever | expected effect on THIS host | numerics / determinism | source(s) | falsified.md interaction |
|---|---|---|---|---|
| **A. Sort-based `index_put_(accumulate=True)` in place of `index_add_`** (no custom code) | Removes the CAS loop, but adds a 1M-key radix sort ×4 layers. Net is UNKNOWN: could land near or below the fp32-atomic 2.43 ms, or above it. Cheapest deterministic A/B | fp32 per destination, one bf16 rounding. **Deterministic** | `TensorAdvancedIndexing.cpp` L988–1003; `Indexing.cu` L661–700, L147–252 | LAW-06: a numerics change inside the pinned bf16 path (the plan's ruling 2). None of the F-rows |
| **B. fp32 agg buffer + `index_add_`** | 6.25 → 2.43 ms per call on the 55 % slice ⇒ ≈ −30 % GPU time per batch; converts ≈ 1:1 while the chain is serial (F-47) | better accuracy. **Not deterministic** (fails a must-read-0 probe) | microbench; `Atomic.cuh`; `cuda_bf16.hpp` | LAW-06 ruling; F-47 (why GPU ms count today) |
| **C. fp32 accumulate expressed as `index_put_`/`scatter_add_` under `torch.compile`** | B's gain + Inductor may fuse gather+relu (11 %) and drop the 256 MB `msg` round-trip. Needs `output_code` to confirm the fusion | fp32. **Not deterministic** (Triton atomics); deterministic mode forces fallback | `lowering.py` L4066–4133; `utils.py` L3215–3257 | F-21's order ("torch.compile first") favours trying C before D |
| **D. dst-sorted fused segment kernel (Triton `triton_op`), edge table in shared memory** | The floor: no atomics, no `msg`, no edge-linear output. Plausibly ≪ 2 ms per layer. Largest win of any aggregation option | fp32 in registers. **Bitwise deterministic** | tutorial (triton_op); builder `lib.rs` L794–800 | F-21 does **not** block it: F-21 rejected a ragged-batch kernel for the *dense* path, which "does not have" that problem, and scoped itself away from the axis-graph representation. This path has exactly that problem |
| **E. `torch.segment_reduce` on dst-sorted fp32 `msg`** | measured 2.11 ms (vs 6.25). Needs a sort (collate or once-per-batch GPU) and keeps `msg` | fp32 if fed fp32. **Deterministic** | `SegmentReduce.cu` L92–130, L440–520 | LAW-06 ruling |
| **F. Readout `amax` via segment/contiguous reduce** (nodes of a graph are contiguous) | Targets the 7 % slice: bf16 `amax` is CAS on every arch with ≈ 875-way same-row contention | exact (max). **Deterministic** | `Atomic.cuh` L436–441; `ScatterGatherKernel.cu` L63–68; `gnn_v2.py` L52–58 | none |
| **G. CPU/GPU overlap: 2 batches in flight** | Cycle falls from CPU + GPU to max(CPU, GPU). F-47 measured GPU busy 52.8 % of wall with no overlap, so this is the biggest *serving* lever regardless of kernels | none | CUDA semantics (pinned, `non_blocking`); streams doc | F-47 is the premise; F-46: the collate must stay in `py.detach` or the second thread serializes on the GIL |
| **H. Wire shrink: u8 edge code + int32 indices** | ≈ 4× fewer H2D bytes (5 % slice) and less host memcpy on the CPU-bound chain | bit-identical model input | builder `lib.rs` L794–800; `AT_DISPATCH_INDEX_TYPES` | F-47 (CPU ms convert 1:1) |
| **I. Write collate output straight into a reused pinned buffer** | Drops one host memcpy per array per batch. After warm-up allocation is already cached | none | `CachingHostAllocator.h` L156–223, L299–301 | F-47 |
| **J. Intra-op threads 1–4 in the server process** | Avoids oversubscription against 32 workers on 32 threads. Sign UNKNOWN (the ≥ 1 MB `pin_memory()` uses the pool) | none | cpu_threading note; `_torch_docs.py` L9815–9838 | F-47 |
| **K. `mark_dynamic` on N/E in place of `dynamic=True`** | Compile-time/robustness; kernel speed likely ± small | none | dynamic-shapes doc ("Not recommended") | none |
| **L. CUDA graphs with bucketed padding** | ≤ a few %: launch overhead is small next to ≈ 30 ms GPU per batch; padding adds sink traffic | none (padding to sink rows) | CUDA semantics; CUDAGraph Trees L171–187 | none; do after A–G |
| **M. Serving stream priority when trainer shares the process** | Serving kernels jump the *pending* queue; no preemption, copies unaffected | none | `streams.py` L29–33; CUDA guide | F-44 regime note (how much trainer work exists) |
| **N. MPS for a two-process layout** | Concurrency across contexts instead of time-slicing, *if* the container permits the daemon | none | MPS docs | none; UNKNOWN feasibility |
| **O. `-C target-cpu=znver5` (env-only)** | Low single-digit % on the Rust collate at best | none (integer/memcpy code) | Rust RELEASES 1.82; LLVM X86.td | R2/LAW-13: never committed; LAW-09 bench |
| **P. Process-wide NN eval cache** | Every hit skips a whole leaf. Size unknown until the duplicate rate is measured | exact keys: output-neutral | KataGo cfg L368–378; lc0 `shared_params.cc` L63–83 | none; symmetry-canonical keys would change outputs (equivariance spread 0.146) |
| **Q. L2 persistence / FP8 / CPU inference / incremental eval** | ≈ 0 here, for the reasons given in Q4 and "Outside the box" 3, 4, 6 | FP8 conflicts LAW-06 | whitepaper; `ScaledBlas.cpp` L474; builder | LAW-06 (FP8); none else |

**Order the sources support (INFERRED):** G and the aggregation choice are independent. G is the largest
serving lever and needs no numerics ruling. On the aggregation, A is the zero-code deterministic probe, D the
deterministic floor, and B/C the fast non-deterministic pair. The determinism requirement ("must read 0")
leaves A, D or E. F and H are small, exact, and ride alongside.
