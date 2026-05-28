# Corpus Expansion Stages

This plan turns the corpus into an agent-oriented HIP/ROCm optimization training
ground: not a library delegation manual, but a practical reference for writing
custom kernels that beat, match, or narrowly specialize beyond hipBLAS/rocBLAS,
hipBLASLt, Composable Kernel, hipCUB/rocPRIM/hipCUB/rocThrust, MIGraphX, Triton, and framework-generated kernels.

Vendor libraries are still essential. They are baselines, correctness oracles,
profiling targets, implementation references, and extension surfaces. A corpus
entry is strongest when it teaches both:

- how the strongest library path behaves, and
- where a custom kernel can exploit narrower assumptions.

Every stage should produce navigable docs, runnable examples, measured records,
and negative examples when attempts lose. Timing-only records are acceptable
until counter access is available, but they must be labeled as timing-only.

## Stage 0: Philosophy And Navigation

Goal: make the repository easy for agents to enter without absorbing the wrong
lesson that "the answer is always call a library."

Deliverables:

- Update navigation docs to point agents first to custom-kernel competition
  workflows, then to library baselines and extension docs.
- Keep repo-local skills focused on task triage: classify operation, find
  baselines, choose candidate custom strategies, run evidence checks.
- Add or maintain a custom-kernel competition guide covering when custom kernels
  can win: fixed shapes, fusion, layout control, lower launch overhead, relaxed
  generality, custom numerics, architecture-specific instructions, and avoiding
  runtime/framework overhead.
- Add decision tables for common situations:
  - "small fixed shape versus library overhead"
  - "memory-bound transform versus fused custom kernel"
  - "GEMM with epilogue fusion"
  - "reduction/scan where hipCUB is the baseline"
  - "MIGraphX engine path versus plugin"
  - "Triton kernel versus HIP C++ kernel"

Acceptance criteria:

- An agent can start from `docs/RETRIEVAL_MAP.md`, identify the relevant task
  family, find the strongest baseline, and still see at least one concrete
  custom-kernel path to try.
- Library docs say how to use the library as a baseline or extension surface,
  not as a stopping point.

## Stage 1: Measured Microkernels

Goal: build a dense, measured foundation of optimization patterns where the
agent can learn from simple kernels before moving to full operators.

Initial families:

- memory copy, strided copy, transpose, gather/scatter, vectorized load/store
- reductions: sum, max, argmax, min/max pair, block reduction, warp reduction
- scans: inclusive, exclusive, segmented, block scan
- rowwise kernels: softmax, log-softmax, LayerNorm, RMSNorm, top-k sketches
- elementwise fusion: bias, activation, residual, dropout mask, quant/dequant
- histograms, atomics, warp-aggregated atomics, contention microbenchmarks
- stencil, shared-memory halo, convolution-style neighborhood loads
- launch overhead, HIP Graphs, persistent-kernel skeletons
- global-to-LDS staging/global-to-LDS staging educational copies where architecture supports them

Each task should include:

- baseline kernel, optimized kernel, and at least one intentionally losing
  variant when useful
- correctness tests with awkward sizes, misalignment, non-powers of two, and
  multiple seeds
- shape sweep metadata, GPU metadata, compiler flags, and build command
- timing-only record now; counter-backed record later
- short "why it won/lost" note written for agents

Acceptance criteria:

- Each microkernel has a corpus task, runnable harness, measured record, and
  guide note.
- At least one negative example exists in every major family so agents learn
  boundaries, not just tricks.

## Stage 2: hipCUB/rocPRIM/hipCUB/rocThrust Competitors

Goal: teach agents how to compete with and borrow from hipCUB/rocPRIM/hipCUB/rocThrust without assuming
hipCUB is always the final answer.

Target comparisons:

- hipCUB DeviceReduce versus custom block/warp reductions
- hipCUB BlockReduce inside custom fused kernels
- hipCUB DeviceScan versus custom scan for fixed-size or fused cases
- hipCUB BlockScan inside larger custom operators
- hipCUB DeviceHistogram versus specialized histogram and atomics strategies
- rocThrust transform-reduce versus fused HIP C++ kernels
- rocThrust sort/reduce_by_key baselines for irregular workloads

Custom win hypotheses:

- fixed small reductions where launch overhead dominates
- fusing transform plus reduce/scan into one pass
- avoiding generic temporary storage, dispatch, and iterator abstraction costs
- exploiting known block sizes, known value types, or known segment lengths
- combining hipCUB block primitives with custom global orchestration

Deliverables:

- Self-contained hipCUB/rocPRIM/hipCUB/rocThrust baseline recipes.
- Custom competitors for each hipCUB baseline, including cases that lose.
- Notes showing when to use hipCUB block/warp primitives as parts inside custom
  kernels.
- A benchmark matrix covering small, medium, large, aligned, misaligned, and
  awkward input sizes.

Acceptance criteria:

- For each hipCUB/rocPRIM/hipCUB/rocThrust task, the agent can answer: "What does hipCUB do well here?",
  "What assumptions can my custom kernel drop?", and "Which shape family is
  worth specializing?"

## Stage 3: GEMM, hipBLAS/rocBLAS, And hipBLASLt Competitors

Goal: make GEMM and matmul-adjacent work a serious competition track instead of
a simple "call hipBLAS/rocBLAS" branch.

Target comparisons:

- naive, tiled, vectorized, and tensor-core custom GEMM
- hipBLAS/rocBLAS `sgemm`, `hgemm`, `bf16`, `int8` baselines
- hipBLASLt algorithm search, layouts, epilogues, workspace sweeps
- batched, strided-batched, and grouped GEMM
- small fixed-shape GEMM and matrix-vector kernels
- GEMM plus bias, activation, residual, quantization, or normalization

Custom win hypotheses:

- tiny or skinny matrices where general GEMM overhead dominates
- fixed dimensions that allow compile-time tiling and unrolling
- fused epilogues that avoid extra memory traffic
- known layouts that avoid transpose/copy overhead
- relaxed numeric requirements or custom accumulation policy
- persistent or grouped scheduling for irregular batches

Deliverables:

- hipBLAS/rocBLAS/hipBLASLt baseline harness with reproducible algorithm search.
- Custom GEMM ladder: scalar, shared-memory tiled, vectorized, tensor-core,
  pipelined, and epilogue-fused.
- Shape taxonomy: square, skinny, tall, batched small, grouped irregular,
  transformer MLP, attention projection, and embedding-like cases.
- Records that include math mode, data type, layout, epilogue, workspace,
  algorithm id when available, and tolerance.

Acceptance criteria:

- Agents can inspect a matmul task and choose between writing a custom kernel,
  using hipBLASLt as a baseline, extending Composable Kernel, or recording a losing
  attempt with a clear reason.

## Stage 4: Composable Kernel Customization And Epilogues

Goal: treat Composable Kernel as both a strong reference implementation and a parts bin
for custom kernels, especially when a full handwritten GEMM is not yet viable.

Target topics:

- CK Tile layout algebra and how it maps logical tensors to memory
- threadblock, warp, and instruction tile choices
- Matrix Core MFMA paths and architecture-specific dispatch
- pipeline stages, async copy, global-to-LDS staging where applicable, and shared-memory layout
- custom epilogues: bias, activation, residual, scaling, quant/dequant,
  normalization fragments, reductions, and visitor patterns
- grouped GEMM and persistent scheduler examples

Custom win hypotheses:

- epilogue or prologue fusion not covered cleanly by stock paths
- fixed shapes that can simplify Composable Kernel generality
- extracting Composable Kernel tiling ideas into a smaller purpose-built kernel
- reducing compile-time or binary-size overhead for one narrow target
- using Composable Kernel as a verified baseline while iterating on handwritten kernels

Deliverables:

- Minimal Composable Kernel examples with exact file pointers into `third_party/composable-kernel`.
- Custom epilogue examples with a clear "what to change next" map.
- Side-by-side records: hipBLASLt, Composable Kernel, handwritten kernel, and losing
  variants.
- Agent notes on reading generated or instantiated Composable Kernel types without
  getting lost.

Acceptance criteria:

- Composable Kernel entries are self-contained enough that an agent can customize an
  epilogue or tile shape, then decide whether to keep Composable Kernel or write a
  narrower custom kernel.

## Stage 5: MIGraphX Plugins And Inference Kernels

Goal: cover inference paths where custom kernels compete with MIGraphX engine
fusion, plugin implementations, vLLM on ROCm kernels, and framework runtimes.

Target topics:

- MIGraphX plugin lifecycle, shape inference, formats, enqueue, serialization
- plugin kernels for fused activation, LayerNorm/RMSNorm, top-k, sampling,
  quant/dequant, rotary embedding, KV-cache update, and attention-adjacent ops
- vLLM on ROCm engine behavior: batching, paged KV cache, quantization,
  decode/prefill separation, and runtime overhead
- engine build versus runtime benchmark discipline
- correctness oracles from framework implementations and MIGraphX outputs

Custom win hypotheses:

- small-batch inference where plugin launch and memory traffic dominate
- fused pre/post-processing around MIGraphX subgraphs
- custom layouts for KV cache, logits processing, or quantized weights
- narrow shape and dtype specialization not worth a generic MIGraphX path
- avoiding framework-to-engine boundary overhead

Deliverables:

- Minimal plugin skeletons with benchmark harnesses.
- MIGraphX and vLLM on ROCm baseline recipes stored separately from custom
  kernel competitors.
- Records separating engine build time, warmup, steady-state latency, and
  throughput.
- Notes on when a plugin is a bridge to a custom kernel rather than the final
  abstraction.

Acceptance criteria:

- Agents can create or modify a plugin, compare it to MIGraphX behavior, and
  identify the next handwritten kernel candidate.

## Stage 6: Framework And Triton Competitors

Goal: help agents compete with PyTorch extensions, OneFlow operators, Triton
kernels, and compiler-generated code while preserving them as references.

Target topics:

- PyTorch C++/CUDA extension anatomy and dispatcher overhead
- OneFlow operator/runtime anatomy and where HIP kernels live
- Triton kernel equivalents for softmax, normalization, matmul fragments,
  reductions, scans, and fused elementwise pipelines
- inspecting generated LLVM IR / AMD GCN ISA where practical
- matching framework semantics: strides, broadcasting, dtype promotion,
  autograd expectations, and edge cases

Custom win hypotheses:

- fixed layout and contiguous-only assumptions
- forward-only inference where autograd is irrelevant
- fused multi-op kernels that avoid framework dispatch
- simpler scheduling than compiler-generated generic kernels
- explicit architecture targeting that Triton/framework code avoids

Deliverables:

- For each framework/Triton task: framework baseline, Triton baseline when
  applicable, HIP C++ custom kernel, correctness oracle, and timing record.
- Notes that separate semantic compatibility from performance specialization.
- Cases where Triton wins, HIP C++ wins, and both lose to a library baseline.

Acceptance criteria:

- Agents can use Triton or framework code as readable reference material while
  still producing and evaluating a custom HIP competitor.

## Stage 7: GPU And GFX Architecture Sweeps

Goal: teach agents that a kernel is not "optimized" in the abstract; it is
optimized for a GPU, GFX generation, dtype, shape, and compiler.

Architecture dimensions:

- gfx target and exact GPU model
- GFX count, memory bandwidth, cache behavior, shared-memory capacity, register
  pressure, and occupancy limits
- available instructions: matrix cores, async copy, global-to-LDS staging, warp-group MFMA,
  architecture-specific atomics, and newer memory paths where supported
- compiler version, target flags, LLVM IR / AMD GCN ISA snippets, and runtime driver

Sweep plan:

- Run every core task on at least two architectures when available.
- Store exact hardware metadata beside every result.
- Maintain architecture notes for sm_70/sm_75, gfx90a/gfx1030, gfx1100,
  gfx942/gfx950, and newer CDNA4/RDNA4 targets as hardware becomes available.
- Add compile-flag sweeps for `-arch`, register caps, fast math, line info,
  maxrregcount, and relevant LLVM IR / AMD GCN ISAAS options.
- Add shape sweeps that expose architecture-specific breakpoints.

Acceptance criteria:

- A record never claims a universal win without architecture scope.
- Agents can ask "what changes on this GPU?" and find both guide text and
  measured evidence.

## Stage 8: Profiler And Counter-Backed Upgrades

Goal: upgrade timing-only observations into profiler-backed explanations once
rocprofiler/rocprof counter access is available.

Counter themes:

- memory throughput, sector/request patterns, coalescing, L2 behavior
- shared-memory bank conflicts
- occupancy, registers, spills, and launch configuration
- warp stall reasons and instruction mix
- tensor-core utilization and eligible warps
- atomic contention and replay behavior
- achieved bandwidth versus theoretical roofline

Deliverables:

- rocprofiler/rocprof collection scripts that degrade gracefully when counters are
  blocked.
- Counter-backed versions of existing timing-only records.
- A profiler triage guide: which metrics to collect for memory-bound,
  compute-bound, launch-bound, atomics-heavy, and tensor-core kernels.
- A record format for attaching short profiler explanations without inventing
  evidence.

Acceptance criteria:

- Timing-only records stay useful but are clearly labeled.
- Counter-backed records explain wins and losses with actual profiler data.

## Stage 9: Automated Eval Harness

Goal: make the corpus continuously expandable and suitable for agent evaluation,
not just human browsing.

Harness capabilities:

- build, run, validate, benchmark, and record one task by id
- run shape sweeps and dtype sweeps
- compare against registered baselines: library, framework, Triton, and prior
  custom kernels
- enforce correctness tolerances and randomized seeds
- emit JSONL records with hardware, compiler, command, timing, and evidence
  level
- support negative examples as first-class results
- detect missing metadata and fail validation

Agent-eval capabilities:

- prompt an agent with a task, baseline, constraints, and target GPU
- run the submitted kernel against correctness and timing checks
- compare against corpus baselines
- store the attempt, patch, explanation, result, and failure mode
- classify wins: absolute win, shape-specialized win, fusion win, launch-overhead
  win, library remains faster, correctness failure, compile failure

Acceptance criteria:

- Adding a new task requires a small manifest plus source files, not bespoke
  scripts.
- The harness can evaluate both human-written and agent-written kernels under
  the same evidence discipline.

## Stage 10: Continuous Expansion Loop

Goal: keep expanding after the initial stages without losing structure.

Loop:

1. Pick a task family and strongest known baseline.
2. Add or verify the library/framework/Triton baseline.
3. Add one simple custom kernel and one stronger custom kernel.
4. Add at least one shape where custom should plausibly win.
5. Run correctness and timing records.
6. Add a short guide note explaining the outcome.
7. Add a negative example if the first hypothesis fails.
8. Promote the task into architecture sweeps and profiler-backed records.

Priority order:

- launch-bound and fusion-heavy kernels where custom kernels are likely to win
- reduction/scan/normalization families that recur inside larger operators
- GEMM epilogue and small-shape matmul cases
- inference plugin and runtime-bound workloads
- architecture-specific instruction and memory-path examples

Done means the agent learns a reusable optimization boundary, not merely that a
single kernel was faster once.
