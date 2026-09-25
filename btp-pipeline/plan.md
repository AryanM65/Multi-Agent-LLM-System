# Dataset Formation Plan — BTP-I Uncertainty Propagation Dataset

> Companion to `correct_project_context.md` (project state) and the original master fix-plan. This document is the execution plan for the actual dataset-generation run: topologies, faults, backend choice, and the Kaggle CLI mechanics. Status: topology pool done, gpt-oss/Ollama pilot done, real Qwen2.5-7B-Instruct-AWQ calibration done (n=3, see `correct_project_context.md` Section 10). Remaining: `scripts/run_study_vllm.py` (the batched generation script) and its smoke test.

---

## 0. The vLLM question, answered directly

**Short answer: vLLM cannot serve `gpt-oss:20b` on Kaggle's free-tier GPUs — this is a hard, verified incompatibility, not a tuning problem. Decision made: switch models and use vLLM.**

`gpt-oss-20b`'s native quantization is MXFP4, and vLLM's MXFP4 kernels for this model (Marlin MXFP4 MoE) require **Ampere or newer** GPU architecture (compute capability ≥ 8.0 for the Triton/Marlin fallback path; some sources state ≥9.0/Hopper for fully-native MXFP4 tensor-core support). Kaggle's free GPU pool is:

| Kaggle GPU | Architecture | Compute capability | MXFP4-on-vLLM compatible? |
|---|---|---|---|
| Tesla T4 (x2, 16GB each) | Turing | SM75 | **No** |
| Tesla P100 (16GB) | Pascal | SM60 | **No** |

There is an active vLLM community forum thread ("Build issues when serving gpt-oss-20B on Tesla T4 GPUs with vLLM") documenting exactly this failure. This is not something a config flag fixes — it's a kernel/architecture gap.

Three options were weighed: (A) keep Ollama Cloud as-is, (B) run Ollama itself (not vLLM) inside a Kaggle GPU kernel, (C) switch to a smaller, standard (non-MXFP4) model that vLLM supports on T4. **Decision: Option C**, chosen model **`Qwen/Qwen2.5-7B-Instruct`**. Rationale:

- **Real batching throughput**: vLLM's continuous batching gives a genuine speed multiplier for a ~400-trial run (thousands of individual node calls) — neither A nor B gets this, since both Ollama paths serve one request at a time.
- **Fits, but not comfortably at fp16 — use a quantized build instead.** Qwen2.5-7B-Instruct is ~7.62B params; at fp16 that's **~15.2GB of weights alone** on a 16GB T4, leaving only ~0.8GB for CUDA context, activations, and KV cache combined. Batching throughput is directly gated by how much KV cache room is available, so this would starve the exact thing vLLM is being adopted for — and `gpu_memory_utilization=0.85` reserving less than the weights need can fail as an outright OOM at model-load time, not just run slow. **Decision: use `Qwen/Qwen2.5-7B-Instruct-AWQ`** (4-bit AWQ quantization, vLLM native support via `quantization="awq"`), dropping weight memory to ~4-5GB and leaving ~10GB+ genuinely free for KV cache/batching — this is the real fix, not a config tweak to try later. (T4 is also Turing — no native bf16 tensor-core support like Ampere+, so serve in **fp16** compute dtype regardless of the weight quantization.) Fallback if AWQ quality is a concern: Kaggle's dual-T4 option with `tensor_parallel_size=2` splits the fp16 model across both GPUs (~32GB combined), sidestepping the headroom problem without quantizing — worth keeping in back pocket if AWQ's accuracy loss turns out to matter for this task, but AWQ is the starting default since it's simpler (one GPU, no tensor-parallel setup) and 4-bit AWQ quality loss is typically small for a fairly extractive/format-following task like this pipeline's node roles.
- **No hidden reasoning pass**: Qwen2.5-Instruct is a standard instruction-tuned model, not a reasoning model — it has no separate `thinking` field competing for the token budget. This **eliminates the entire empty-sample/thinking-fallback bug class** that consumed a large fraction of this session's debugging on gpt-oss. `used_thinking_fallback` tracking becomes dead code for this model (harmless to leave in place — it'll just always be `False` — but not load-bearing).
- **Strong instruction-following**: the pipeline depends on strict format compliance (`FINAL ANSWER: <answer>`, "output only sentences, no commentary") — Qwen2.5-Instruct is one of the more reliable open 7-8B models at this specifically.
- **Real cost, accepted knowingly**: every piece of gpt-oss-specific calibration from this session (`MAX_TOKENS`/`MAX_TOKENS_REASONER` values, `NODE_THRESHOLDS`) needs to be redone for Qwen2.5's own baseline behavior — Section 0.1 below is the recalibration plan. This is bounded, well-understood work now (the tooling — `verify_results.py`, the debug scripts, the unit tests — already exists and just needs re-pointing at new data), unlike when it was first done from scratch.

Options A and B are kept below (Section 0.2) as a documented fallback only, in case Kaggle GPU access itself becomes unavailable (quota exhausted, account issue) and a return to Ollama Cloud is needed as a stopgap.

### 0.1 Recalibration plan for Qwen2.5-7B-Instruct (must happen before bulk generation)

This mirrors the original Phase 2 gpt-oss calibration, using the tooling already built this session:

1. **Token budget calibration**: run a small pilot (2-3 questions, k=5, clean control only) and measure real output-token usage via vLLM's own token counts (no hidden-reasoning complication here, so this should be simpler and more predictable than the gpt-oss case). Set `MAX_TOKENS`/`MAX_TOKENS_REASONER` with sensible margin above the observed max, not copied from the gpt-oss values. **Run this calibration against the actual deployed backend — `Qwen/Qwen2.5-7B-Instruct-AWQ` loaded via vLLM with `quantization="awq"` — not a full-precision build of Qwen2.5 assumed to transfer.** Quantization can shift token-level output behavior slightly; calibrating against a different build than what generation actually runs on would reintroduce the exact "estimate doesn't match reality" failure mode this project has already been burned by twice.
2. **`FINAL ANSWER:` marker compliance check**: confirm Qwen2.5 reliably follows the Reasoner's marker instruction across several samples — `extract_conclusion()` already handles the marker-present case cleanly and falls back to last-line otherwise, but verify the fallback path is rarely needed (unlike the thinking-fallback problem, this fallback has no separate-field complication, so a bad case is just a slightly-off conclusion, not an empty string).
3. **`parse_retrieved_items` sanity check**: re-run the "10-15 real outputs" manual check (per the original master plan's Fix 2.3 verification step) against Qwen2.5's actual Retriever-role outputs — the newline-first parser should generalize, but confirm on real data rather than assuming.
4. **Achievable-uncertainty-value-set check at k=5**: confirm lexical/semantic uncertainty spans the expected 6-level set `{0, 0.2, 0.4, 0.6, 0.8, 1.0}` on clean baseline data for this model (same check the original plan required after raising k, now due for the new model too).
5. **`NODE_THRESHOLDS` recalibration**: set per-node thresholds from Qwen2.5's own clean-baseline uncertainty distribution (retriever/reasoner/writer will each have a different natural baseline, as gpt-oss did) — do **not** carry over the gpt-oss-tuned values (`0.25`/`0.50`/`0.25`) without re-deriving them.
6. **Jaccard unit test stays valid as-is** — `tests/test_uncertainty.py` tests the math itself, not model-specific behavior, so no changes needed there.

Since diagnosis is disabled for the bulk run (`--no-diagnose`, Section 3), steps 2 and 5 are lower-priority than 1/3/4 for the dataset-generation path specifically — but should still be done before generation, since `NODE_THRESHOLDS`/marker-quality still affect what counts as a "clean" pilot to validate against, and will matter immediately if the retry-then-reprobe protocol is ever evaluated later as the naive-baseline comparator.

### 0.2 Fallback options (not the current plan — documented for reference only)

**Option A — Keep Ollama Cloud (gpt-oss:20b, current setup)**: zero infrastructure work, everything already calibrated, but no batching and blocked on the (still-unchecked) Ollama Cloud quota fitting the ~4.5-5x cost increase.

**Option B — Run Ollama itself (not vLLM) inside a Kaggle GPU kernel**: Ollama's `ggml`/`llama.cpp` kernels tolerate older hardware (T4) better than vLLM's MXFP4 path, but still no batching advantage, and the install-inside-a-Kaggle-kernel step has never been tried in this project.

Return to this section only if the Qwen2.5+vLLM path (Section 0's decision) hits a blocker that isn't a simple retry (e.g., Kaggle GPU quota fully exhausted with no time to wait for reset, or vLLM itself has an unexpected T4 compatibility issue with Qwen2.5 specifically — unlikely given it's one of the most common vLLM benchmark models, but not zero-risk).

---

## 1. Prerequisites (must close before Step 2 begins)

Updated after the Section 0 decision to switch to Qwen2.5-7B-Instruct + vLLM — the Ollama-Cloud-specific quota check is no longer a blocking gate for this plan (it only matters for the Section 0.2 fallback path):

1. **(Non-blocking, downgraded)** gpt-oss/Ollama combined-fix pilot (`logs/pilot_combined_v2.jsonl`, in progress at time of writing) — finish if it's cheap to let run out, but **do not let it hold up starting Section 0.1's Qwen2.5 recalibration.** Its original purpose was validating the reasoning-model empty-content/thinking-fallback fix, which is now structurally irrelevant — Qwen2.5-Instruct isn't a reasoning model and won't hit that bug class. The other thing it exercises (topology engine, fault injection, the `build_prompt` regression fix) is already validated via mock-mode tests and the `debug_topology_regression.py`/`debug_noise_scoping.py` executed tests, independent of which model backend is used.
2. **Qwen2.5-7B-Instruct recalibration** (Section 0.1) — token budgets, marker compliance, `parse_retrieved_items` spot-check, k=5 achievable-value-set, `NODE_THRESHOLDS`. This replaces the old "check Ollama Cloud quota" gate as the actual blocker for this path.
3. **Topology pool generated and frozen** (Step 2 below) — this is itself part of this plan, not a separate gate, but must be done and the file committed *before* any trial generation starts, per the master plan's explicit requirement that the train/OOD split be fixed in advance.
4. **`--no-diagnose` confirmed working against a real backend run** (currently only verified in mock mode) — the combined pilot in item 1 can double as this check if it's re-run with `--no-diagnose` once item 1's diagnosis-on run is done being useful for accuracy comparison. Should also be spot-checked once the vLLM path exists, since it's new code (Section 4.3).

---

## 2. Topology pool formation — DONE

Implemented as `dataset/generate_topology_pool.py` (in the top-level `dataset/` folder, sibling to `btp-pipeline/`, not inside `btp-pipeline/scripts/`). Output frozen to `dataset/topology_pool.json`: 14 train + 6 OOD topologies, validated, deduplicated, round-trip-loadable. The design below (originally written before this was built) matches what was actually implemented; kept as the design record.

### 2.1 Train pool (~14 topologies)

Built from `src/topologies.py`'s existing factories plus hand-authored variants:

| Category | Count | Source |
|---|---|---|
| 3-node chain (baseline) | 1 | `chain_topology()` |
| Dual-retriever fan-in variants | 2-3 | `dual_retriever_fanin_topology()` + hand variants (differ in which retriever's output is used first / instruction wording) |
| Fan-out (parallel reasoners) variants | 2-3 | `parallel_reasoner_topology()` + variants |
| Deep chain (4+ nodes) | 2-3 | `deep_chain_topology()` + a 5-node variant |
| Named canonical shapes (star, tree) | 2-4 | New — see 2.3, informed by the "Topology Matters" paper's canonical shape taxonomy referenced in the project context |

Every topology **must pass `validate_role_order()` and `topological_order()` without error** before being added to the pool — this is already enforced by `src/topology.py`, just confirm it's called during pool construction, not just at pipeline-run time.

### 2.2 Held-out OOD pool (~4-6 topologies)

Generated via **random-DAG construction**, MOC-style, exactly as specified in the original master plan:
1. Pick a random topological order over a chosen node count (5-7 nodes for OOD, deliberately larger/more complex than most train-pool topologies, to actually test structural generalization).
2. Assign roles to positions **before** adding edges, respecting `ROLE_RANK` (retriever positions first, writer positions last) — this guarantees every generated topology is role-valid by construction, never post-hoc filtered.
3. Force a connected backbone chain across the ordered nodes (so there are no disconnected components).
4. Randomly add additional forward-only edges (respecting the fixed topological order) with some probability (e.g. 30%) to create fan-in/fan-out structure beyond the plain backbone.
5. Deduplicate against both the OOD pool itself and the train pool (compare by canonical edge-set + role-assignment, not just `topology_id` string) — reject and regenerate on collision.

Implemented in `dataset/generate_topology_pool.py` (see Section 2 header — done, not still needed).

### 2.3 Freezing the pool

Output: `dataset/topology_pool.json`, containing every topology's full `NodeSpec`/edge definition (not just a name — the pool is reproducible without re-running the generator via the included `load_topology_pool()` helper), tagged `"split": "train"` or `"split": "ood_test"`. This file is committed and should not be regenerated once trial generation begins against it — regenerating it after starting generation would silently invalidate the train/OOD split guarantee the whole design depends on.

---

## 3. Fault condition coverage

Per topology, faults are drawn from the existing `build_fault_conditions(topo)` grid — `{noise, contamination, ceiling} × topo.nodes.keys()`, generalized automatically to however many nodes a given topology has (already implemented, confirmed working across all three fault types × all three roles this session).

**Coverage strategy** (unchanged from the original master plan, restated for clarity):
- **Full grid** (control + every fault type × every node) on the simplest 3 topologies (plain chain, one fan-in, one fan-out) — gives a complete, dense core dataset for the most structurally-simple cases.
- **Partial coverage** (control + at least one fault type per node, not the full cross-product) on every other topology — maximizes topology diversity within the compute budget rather than exhaustively re-testing the same fault/node combinations on structurally similar topologies.

**Diagnosis stays disabled** (`--no-diagnose`) for all of this — per the decision already made this session, since the dataset's consumers (belief propagation, GNN) need raw per-node uncertainty + verified `true_label`, not the retry-heuristic's own diagnosis.

---

## 4. Generation procedure

### 4.1 Trial record shape (unchanged from current schema)

Every trial already logs: `question`, `true_label`, `topology_id`, `uncertainties` (lexical, per node), `semantic_uncertainties`, `samples` (raw, per node, per k), `used_thinking_fallback` (per node, per k — new this session), `conclusions` (Reasoner), `jaccard_uncertainties`/`item_frequencies` (Retriever, multi-item case), `gold_answer`, `verification` (z-score vs. per-question baseline). This is already the right shape for GNN training data (node features + verified label) — no schema change needed for dataset generation itself, only for consumption (a later, separate step converts trial JSONL → graph-structured training examples, out of scope for this plan).

### 4.2 Per-topology loop

For each topology in the frozen pool:
1. Assign it its share of questions (per the train/OOD split — OOD topologies only ever see the held-out question set, never touched during any train-pool generation).
2. Run the clean control trial **first**, per question — establishes that question's baseline for `verify_against_baseline`.
3. Run the topology's assigned fault conditions (full grid or partial, per Section 3).
4. Append every result to an append-only JSONL immediately (already implemented — crash-safe, resumable via `--resume` / `done_keys`).

### 4.3 vLLM batching (the chosen path — this is new code, not yet written)

This is the main implementation task remaining after the topology pool. `run_study1.py`/`run_study2.py`'s current design calls `run_pipeline()` once per trial, sequentially — correct for Ollama's one-request-at-a-time API, but wasteful for vLLM, whose entire value proposition is batching many prompts into one `generate()` call.

**New entry point needed**: `scripts/run_study_vllm.py` (or a `--backend vllm` mode added to the existing runners — a design choice to make when writing it, not decided here). Core pattern, adapting the reference implementation already sketched in this project's history:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct-AWQ", quantization="awq", dtype="float16",
          gpu_memory_utilization=0.85, max_model_len=4096)

def run_batch_of_trials(trials: list, topo, llm, k: int, temperature: float):
    order = topological_order(topo)
    outputs_per_trial = [dict() for _ in trials]
    for node_id in order:
        stage_prompts, stage_params = [], []
        for i, trial in enumerate(trials):
            prompt = build_prompt(topo, node_id, outputs_per_trial[i],
                                    trial["question"], trial["context"], trial["fault_config"])
            node_temp = temperature
            if trial["fault_config"] and trial["fault_config"].get("target_node") == node_id \
                    and trial["fault_config"]["type"] == "noise":
                node_temp = NOISE_TEMPERATURE
            stage_prompts.append(prompt)
            stage_params.append(SamplingParams(temperature=node_temp, max_tokens=MAX_TOKENS, n=k))
        stage_outputs = llm.generate(stage_prompts, stage_params)
        for i, output in enumerate(stage_outputs):
            samples = [c.text for c in output.outputs]
            result = compute_node_result(node_id, topo.nodes[node_id].role, samples)  # reuses existing uncertainty logic from src/nodes.py / src/uncertainty.py
            outputs_per_trial[i][node_id] = result.output
            trials[i].setdefault("node_results", {})[node_id] = result
    return trials
```

Key points:
- **`quantization="awq"` on `Qwen2.5-7B-Instruct-AWQ`, not the full-precision model** — see Section 0's VRAM analysis: fp16 weights alone (~15.2GB) leave almost no room for KV cache on a 16GB T4, which would starve the batching throughput this whole switch is for, and can OOM at load time outright. AWQ drops weights to ~4-5GB, leaving real headroom.
- **`dtype="float16"`, not `"bfloat16"`** — T4 lacks native bf16 tensor-core support (Turing architecture); this is the one Qwen2.5-specific vLLM config detail that matters for correctness/speed on this GPU, independent of the quantization choice above.
- Batch **within one topology at a time** (a batch assumes one shared `topo` for `build_prompt`/`parents_of` calls) — group trial specs by `topology_id` before batching, per the existing per-topology loop structure (Section 4.2).
- Batch size ~15-30 trials at a time, not the whole run at once — keeps memory bounded and allows checkpointing between batches (append results to the JSONL after each batch completes, not only at the very end). Confirm the actual safe batch size empirically during the Section 5.5 smoke test rather than assuming 15-30 fits — it depends on realized KV-cache headroom, which the smoke test now explicitly checks.
- Reuse `compute_node_result`-equivalent logic from `src/nodes.py`/`src/uncertainty.py` for the actual uncertainty computation once raw samples are back — **do not reimplement lexical/semantic/Jaccard uncertainty math for the vLLM path**; only the generation call itself changes.
- Fault application (`apply_fault_to_prompt`, `build_prompt`'s corruption hooks) is unchanged — these operate on prompt text and are backend-agnostic.
- **New risk specific to this batched code, not present in the old sequential design**: `run_batch_of_trials` processes multiple trials with potentially *different* `fault_config`s in a single `llm.generate()` call, matching each prompt to its own `SamplingParams` positionally (e.g. one trial's noise fault sets `node_temp = NOISE_TEMPERATURE` for its prompt while another trial in the same batch keeps `temperature` at default). This should work correctly since each prompt/params pair is independent, but it is untested new code doing exactly the kind of per-trial-scoped-state handling that already produced one silent bug this session (the `build_prompt` continuation-cue regression, Section 9.4 of `correct_project_context.md`) — see Section 5.5's smoke test, which now explicitly checks this rather than assuming it's fine because it "looks obviously correct."

**Not yet written.** The topology pool (Section 2) is done and the Qwen2.5 calibration pilot (Section 0.1) has run once (n=3 questions, real results in `correct_project_context.md` Section 10.2 — token budgets confirmed sufficient, no changes needed there). This is now the single concrete next implementation step.

---

## 5. Kaggle CLI workflow (the chosen path — Qwen2.5-7B-Instruct + vLLM)

### 5.1 One-time setup
```bash
pip install kaggle
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

### 5.2 Package the codebase
```bash
cd btp-pipeline
kaggle datasets init -p .   # edit dataset-metadata.json: title, id, license
kaggle datasets create -p . # re-run `kaggle datasets version -p .` on later code changes
```
Include `requirements.txt` with `vllm` pinned to a version confirmed compatible with T4 (check vLLM's release notes for Turing/SM75 support on whichever version is current when this is run — this project's earlier research confirmed *general* T4 support in vLLM for standard fp16 models; it was specifically MXFP4/gpt-oss that failed, not vLLM-on-T4 broadly).

### 5.3 Kernel scaffolding
```bash
kaggle kernels init -p .
# edit kernel-metadata.json:
#   enable_gpu: true
#   enable_internet: true          # needed to pull the Qwen2.5-7B-Instruct weights from HF Hub on first run
#   dataset_sources: ["your-username/btp-pipeline-code"]
#   code_file: "kaggle_run_study_vllm.py"   # the vLLM entry point from Section 4.3
```

### 5.4 Push, run, retrieve
```bash
kaggle kernels push -p .
kaggle kernels status your-username/btp-generation-run
kaggle kernels output your-username/btp-generation-run -p ./results
```

Checkpoints go to `/kaggle/working/` (the kernel's downloadable output). A script kernel is single-shot/non-interactive — recovery from a crash or timeout is pulling whatever JSONL was written and pushing a follow-up run that resumes via `done_keys` (already implemented, reused as-is for the vLLM path). GPU time drawn this way is the **same 30-hour weekly quota** as interactive notebook sessions, not a separate allowance — budget accordingly against the Section 0.1/original Section 9.1 cost estimates (note: those specific 4.5-5x/hours-to-a-day numbers were measured for gpt-oss on Ollama Cloud and do **not** directly transfer to Qwen2.5+vLLM — re-measure wall-clock during the Section 0.1 recalibration pilot instead of assuming the old numbers apply).

### 5.5 First-run smoke test (do this before trusting a multi-hour kernel run)

Before pushing a long generation kernel, push a short smoke-test kernel first, checking specifically:

1. **GPU actually in use**: check kernel logs for the CUDA device vLLM reports initializing on — not a silent CPU fallback.
2. **Model loads without OOM, and the realized safe batch size is checked, not assumed.** Run `run_batch_of_trials` with a small batch (2-3 questions, one topology, full grid) and confirm `LLM(...)` initializes cleanly at `quantization="awq"`. Then deliberately push batch size up (e.g. try 15, then 30) and watch for OOM or a large latency cliff — the 15-30 figure in Section 4.3 is a starting guess, not a verified number; this step is what actually verifies it, replacing the "hoped to be fine" status from before this VRAM analysis was added. While doing this, also sanity-check **realized throughput**, not just "didn't OOM": each prompt's `k=5` completions (`n=5` in `SamplingParams`) share the prompt's KV cache via vLLM's prefix caching, so per-call cost is cheaper than 5 fully independent sequences — confirm the batch's actual wall-clock looks like a meaningful multiple of sequential Ollama-style generation, since that speedup is the entire reason for this backend switch, not an assumed side effect of it.
3. **Batched noise-scoping check**: batch 2-3 trials together in one `run_batch_of_trials` call where only one trial has `fault_config={"type": "noise", "target_node": ...}` and the others are clean/different fault types. Confirm only the noise trial's targeted node actually received `NOISE_TEMPERATURE` in its `SamplingParams` — i.e. that the positional prompt-to-params matching in the new batched code path isn't silently misaligning temperatures across trials. `debug_noise_scoping.py` (existing) only covers the old sequential single-trial path; this is new coverage for the batched path specifically, since it's new code handling exactly the kind of per-trial-scoped state that already caused one silent bug this session.
4. **Output quality and schema**: confirm output quality looks sane, and confirm the JSONL checkpoint format matches what `verify_results.py` expects.

This is a cheap way to catch a vLLM/T4/Qwen2.5 integration problem in minutes instead of discovering it after a multi-hour kernel run.

---

## 6. Verification during and after generation

- `verify_against_baseline` already runs inline per trial (no change needed).
- After generation (or after each Kaggle kernel batch, if resumable multi-session), run `scripts/verify_results.py <log> --skip-log <skip_log>` — recompute detection/coverage stats fresh, never trust a prose summary of a generation run (this project's own repeated failure mode, per `correct_project_context.md` Section 1).
- Spot-check `used_thinking_fallback` rate and the fallback word-count distribution (already surfaced by `verify_results.py`) on every batch — if a new topology or question type starts producing a much higher fallback rate than the pilot did, investigate before continuing rather than generating hundreds more contaminated trials.
- Confirm the final trial count matches the intended split (~240 train / ~40 validation / ~120 OOD) before considering generation "done" — a shortfall from skips (contamination-insufficient-distractors, ceiling-answer-survived) should be backfilled with additional questions on the same topologies, not silently accepted as a smaller dataset.

---

## 7. Definition of done for this phase

- [x] *(Optional, non-blocking)* gpt-oss/Ollama combined-fix pilot (`logs/pilot_combined_v2.jsonl`) — completed
- [x] `dataset/generate_topology_pool.py` written, run, output frozen to `dataset/topology_pool.json`, committed (14 train + 6 OOD)
- [x] **Qwen2.5-7B-Instruct recalibration (Section 0.1) — partial, n=3 questions**: token budgets confirmed sufficient from real measurement (0/45 empty samples), marker-compliance confirmed (15/15), `parse_retrieved_items` spot-checked clean on real output. **Not done**: `NODE_THRESHOLDS` re-derivation (deferred — not needed for generation since diagnosis is disabled; n=3 is too small a sample anyway, see `correct_project_context.md` Section 10.2). Recommend a larger calibration pass (~15-20 questions) before ever trusting thresholds for Qwen2.5.
- [ ] `scripts/run_study_vllm.py` (or equivalent) written per Section 4.3 — **the current next concrete task**
- [ ] Kaggle smoke-test kernel (Section 5.5) passes — GPU confirmed in use, output format confirmed compatible with `verify_results.py`
- [ ] Full generation run completes (or resumes cleanly across multiple Kaggle sessions if needed)
- [ ] `verify_results.py` run on the final combined log, numbers match the intended split sizes
- [ ] Fallback-contamination check: `used_thinking_fallback` should be near-universally `False` for Qwen2.5 (it's not a reasoning model) — if it's not, that's itself a signal something unexpected is happening and worth investigating before trusting the dataset

Ollama Cloud quota checking (previously listed here) is no longer a blocking gate for this plan's chosen path — it only matters if Section 0.2's fallback is ever invoked.

---

## Sources (vLLM/GPU compatibility claims in Section 0)

- [Build issues when serving gpt-oss-20B on Tesla T4 GPUs with vLLM — vLLM Forums](https://discuss.vllm.ai/t/build-issues-when-serving-gpt-oss-20b-on-tesla-t4-gpus-with-vllm/1673)
- [vLLM Now Supports gpt-oss — vLLM Blog](https://blog.vllm.ai/2025/08/05/gpt-oss.html)
- [GPT OSS — vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/stable/OpenAI/GPT-OSS.html)
- [openai/gpt-oss-20b — MXFP4 only runs on h100 or b100 or later versions (HF discussion)](https://huggingface.co/openai/gpt-oss-20b/discussions/61)
- [Hardware Requirements for Running GPT-OSS-20B Locally — IntuitionLabs](https://intuitionlabs.ai/articles/hardware-requirements-gpt-oss-20b)
- [Differences Between NVIDIA GPU T4 and NVIDIA GPU P100 — Kaggle](https://www.kaggle.com/getting-started/561774)
- [Weekly Maximum GPU Usage — Kaggle](https://www.kaggle.com/general/108481)
