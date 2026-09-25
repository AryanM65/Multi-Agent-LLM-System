# Project Context — Multi-Agent LLM System: Uncertainty Propagation Pipeline

> **BTech Bachelor's Thesis Project (BTP-I)** — Aryan M, 2026
> Last verified against actual code (not prose docs) on 2026-09-24, branch `corrected-pipeline`.

---

## 1. What This Project Is

This is a **research codebase** for a BTech thesis investigating a specific question:

> *When one agent in a multi-agent LLM pipeline fails, can we automatically detect WHICH agent failed and WHY — using only the model's own sampling behaviour as a signal, without any external oracle or ground-truth checker?*

Modern AI systems frequently chain multiple LLM agents: a **Retriever** fetches evidence, a **Reasoner** interprets it, and a **Writer** synthesises the final answer. Each agent looks locally correct — but if the Retriever passes bad evidence, the Reasoner reasons *perfectly from bad inputs* and the Writer produces a confident-sounding but wrong final answer.

**Standard systems only inspect the final output.** This project builds a mechanism that watches *every intermediate node* using self-consistency sampling and diagnoses failures at their root cause — and, as of this branch, does so across **arbitrary DAG topologies**, not just a fixed 3-node chain.

---

## 2. How This Doc Differs From the Previous Version

The prior version of this file described the state as of the threshold-bug fix only (a fixed 3-node chain, Retriever-only faults, k=3, no verification tooling). A subsequent code audit (grepping/reading `src/*.py` directly rather than trusting prose) found that **most of a much larger planned redesign — topology-generalization, node-targeted fault injection, reproducible seeding, per-question baseline verification — has actually already been implemented in code**, even though no doc described it. This version reflects that direct code inspection. Two items remain genuinely undone (Section 8).

---

## 3. Core Architecture — Now Topology-Generic

### 3.1 The Topology Engine (`src/topology.py`, `src/topologies.py`)

The pipeline is no longer a hardcoded Retriever→Reasoner→Writer chain. `src/topology.py` defines:

- `NodeSpec(node_id, role, instruction)` — role ∈ {retriever, reasoner, writer}
- `Topology(nodes, edges, topology_id)` — a DAG of nodes
- `validate_role_order(topo)` — rejects edges that violate retriever→reasoner→writer ordering
- `topological_order(topo)` — Kahn's-algorithm topological sort, raises on cycles
- `parents_of(topo, node_id)` / `children_of(topo, node_id)`
- `default_chain_topology(...)` — the original 3-node chain, now just one instance of the general engine

`src/topologies.py` adds concrete topology factories used for multi-topology study runs:
- `chain_topology()` — 3-node baseline
- `dual_retriever_fanin_topology()` — two retrievers feeding one reasoner
- `parallel_reasoner_topology()` — fan-out reasoning
- `deep_chain_topology()` — 4+ node chain
- `get_topology(name)` / `all_topologies()` — registry/lookup helpers

`src/pipeline.py`'s `run_pipeline()` takes a `Topology` object, computes `topological_order`, and builds each node's prompt from its actual parents via `parents_of` — there is no node-count assumption baked into the execution loop anymore.

### 3.2 The Uncertainty Measures

Each node is sampled **k times** independently at the same temperature (currently **k=3 — see Section 8**, the plan's target of k=5 is not yet applied).

Three separate uncertainty metrics are computed and stored **without overwriting each other** on `NodeResult` (`src/nodes.py`):
- `uncertainty` — lexical (exact-match based)
- `uncertainty_semantic` — embedding-based (`all-MiniLM-L6-v2`), used for Reasoner/Writer diagnostic triggering
- `uncertainty_jaccard` — set-overlap based, used for Retriever nodes that return multiple discrete items (`src/uncertainty.py`: `jaccard_uncertainty()`, `per_item_inclusion_frequency()`), only computed when a sample yields >1 parsed item

**Why semantic replaced lexical as the trigger (the original branch-defining fix):** at k=3, lexical uncertainty over 3 fully-different samples caps at `1 - 1/3 = 0.667`, but the old global threshold was `0.75` — mathematically impossible to trigger, causing 0.0% detection. Semantic uncertainty compares *meaning* via embeddings, giving a continuous score where paraphrases (e.g., "Yes, both are American" vs. "Yes, they share the same nationality") score near 0.0 despite being lexically different.

**Calibrated per-node thresholds** (all sit below the k=3 achievable ceiling, unlike the original bug):
- `retriever`: lexical/Jaccard, threshold `0.25`
- `reasoner`: semantic, threshold `0.50` (CoT reasoning naturally fluctuates more)
- `writer`: semantic, threshold `0.25`

### 3.3 Reasoner Conclusion Extraction

The Reasoner's prompt (`pipeline.py`) instructs the model to end its response with `FINAL ANSWER: <one-sentence conclusion>`. `uncertainty.py`'s `extract_conclusion()` does a clean regex split on that marker (falling back to the last non-empty line if absent), and semantic uncertainty is computed on the *extracted conclusion*, not the raw chain-of-thought. The old keyword-heuristic extractor (`retroactive_extract_conclusion()`) is kept, but deliberately isolated for retroactive use on old marker-less logs only — it is not used for new data.

### 3.4 Retriever Item Parsing

`parse_retrieved_items()` no longer blindly splits on `re.split(r"[,\n]", ...)` (which used to shred sentences like *"a 4,000 capacity (3,677 seated) arena"* into three garbage fragments on the internal numeric commas). It now splits on newlines first, and only falls back to comma-splitting if the newline split yields ≤1 item — otherwise treats the string as one item.

---

## 4. Fault Injection — Node-Targeted, Not Retriever-Only

Implemented in `src/faults.py` + `src/config.py`.

**Taxonomy** (`config.py`): `FAULT_TYPES = ["noise", "contamination", "ceiling"]` × `TARGET_NODES = ["retriever", "reasoner", "writer"]` (or all nodes in a given topology). `build_fault_conditions(topo)` generates the full grid (1 clean control + up to 9 fault conditions), replacing the old Retriever-only fault set.

1. **Noise** — temperature elevated to `1.2` (`NOISE_TEMPERATURE` in `config.py`), applied only to the targeted node via `apply_fault_to_prompt`, verified in `run_pipeline` to only affect that node's generation call (other nodes keep `DEFAULT_TEMPERATURE = 0.7`). Expectation: recovers on same-input retry at normal temperature.
2. **Contamination** — for the Retriever: `inject_contamination_retriever()` does a *proportional* swap (`swap_fraction`, not the old fixed count of 2), with optional embedder-based plausibility ranking (`_pick_plausible_distractor`, top-k cosine similarity) instead of pure random distractor choice. For Reasoner/Writer targets: `inject_contamination_downstream()` substitutes a plausible-but-wrong output drawn from other questions. Expectation: does not recover on same-input retry; does recover on clean-context retry.
3. **Ceiling** — `_inject_ceiling_retriever()` strips sentences from gold context, but **verifies the answer span does not survive** in the stripped text before logging it as a ceiling trial; if the answer would still be findable, the trial is skipped (`"ceiling_answer_survived"`) rather than mislabeled. A downstream variant (`_inject_ceiling_downstream`) hardens Reasoner/Writer instructions (e.g. step/word limits) since those nodes have no context of their own to strip. Expectation: persists through both retry types (genuine capability gap).

**Reproducible seeding**: `make_rng(question_id, fault_type, target_node)` (`faults.py`) derives a seed via `hashlib.md5`, used by every injector — same `(question_id, fault_type, target_node)` always produces byte-identical corrupted input.

**Skip logging**: any injector that returns `None` (contamination with insufficient distractors, ceiling with a surviving answer) is logged to a skip record (`_skip_record()` in `run_study1.py`/`run_study2.py`, `_is_skip: true`) rather than silently dropped or mislabeled — confirmed present on disk as `logs/trials_new_skip.jsonl` (3 records).

---

## 5. The Diagnostic Protocol

Implemented in `src/diagnose.py`.

When a node's uncertainty exceeds its threshold:
```
Step 1 — Same-input retry
    Run k samples with the IDENTICAL prompt.
    If it drops below threshold -> NOISE (instability, not bad input)

Step 2 — Clean-input retry
    Run k samples with the FULL GOLD context.
    If it drops below threshold -> CONTAMINATION (input was the problem)

Step 3 — Neither helped
    -> CEILING (model genuinely cannot answer this)
```

**Per-question baseline verification** (also `diagnose.py`, `verify_against_baseline()`): for every question, a clean control trial runs first and its per-node uncertainties become that question's own baseline (not a global assumed constant). Every subsequent fault trial for that question is z-scored against its own control (`z = (observed - baseline) / assumed_std`, `assumed_std=0.10`, `z_thresh=1.0`) before being counted as a genuine deviation. This is wired into both `scripts/run_study1.py` and `scripts/run_study2.py` (control trial always runs first per question).

---

## 6. Model, Backend, and Runners

- **Primary model**: `gpt-oss:20b-cloud` via Ollama cloud proxy (adopted after local `qwen3:8b` inference crashed the host with OOM: `failed to allocate CPU_REPACK buffer of size 3312451584` — even 2GB models failed to load locally).
- **Mock backend**: `scripts/run_study2.py` supports `BTP_MOCK=1` / `--mock`, which runs the full pipeline (topology engine, fault injection, diagnosis) against synthetic HotpotQA-shaped examples with no model calls — used for structural/regression testing on machines without MLX/a working local backend (e.g. Windows dev boxes).
- **Runners**: `scripts/run_study1.py` (original 3-node-chain study, now running through the generic engine), `scripts/run_study2.py` (multi-topology study using `src/topologies.py`'s factories), `scripts/run_local_debug.py`, `scripts/rescore_study1.py`, `scripts/compare_study1.py` (before/after metric diffing), `scripts/extract_original_examples.py`, `scripts/download_data.py`.

---

## 7. Results So Far

A 66-trial Study 1 pilot run (`logs/trials_new.jsonl`, 63 trial records + `logs/trials_new_skip.jsonl`, 3 skip records) was independently recomputed directly from the raw log (not trusted from prior prose summaries — two earlier summaries in this project's history did not match their logs when checked):

```
Total fault trials: 56 (control trials excluded)
Diagnosed (non-null): 10/56 (17.9%)
Correct among diagnosed: 4/10 (40.0%)

Per-fault-type detection:
  noise:          4/21 detected (2 correct)
  contamination:  2/20 detected (1 correct)
  ceiling:        4/15 detected (1 correct)
```

These numbers **match** the previously-circulated 17.9% / 40.0% figures — that specific claim held up under raw-log re-verification. However, this is a **small pilot (10 diagnosed trials)** — treat it as suggestive, not conclusive, until the full ~400-trial dataset (Section 9 of the original master plan) is generated. The old pre-fix log (`logs/trials.jsonl`, 45 records, 0.0% detection under the lexical-threshold bug) is preserved separately as the "before" baseline, not overwritten.

---

## 8. Status Update (2026-09-24, same session): Both Remaining Gaps Closed

Both items flagged as missing above have now been addressed:

1. **`k` raised 3 → 5.** `src/config.py:53`: `DEFAULT_K = 5`. Both `scripts/run_study1.py` and `scripts/run_study2.py` default their `--k` CLI flag from `DEFAULT_K`, so this takes effect on any future run without further changes. `run_local_debug.py` still hardcodes `k=3` in several calls — that file is a fixed debug/calibration script for the original threshold investigation, not a production runner, so it was left as-is (its k=3 calibration record in Section 3.2 above stays historically accurate). **Not yet done as a follow-up**: `NODE_THRESHOLDS` in `config.py` were calibrated against k=3 clean-baseline data (see comments at `config.py:69-81`); they have not been re-verified against fresh k=5 baseline trials. Re-run the Phase 2 clean-baseline calibration before trusting detection numbers from a k=5 run.
2. **`scripts/verify_results.py` created.** Recomputes total/diagnosed/correct counts, per-fault-type breakdown, and a confusion matrix directly from a raw trials JSONL (plus an optional skip-log for full attempted-trial accounting), exactly per the plan's Section 1.1 spec. Also adds a sanity check for empty Retriever samples. Usage:
   ```
   python scripts/verify_results.py logs/trials_new.jsonl --skip-log logs/trials_new_skip.jsonl
   ```
   Run against the existing 66-trial log, it reproduces the Section 7 numbers exactly (56 fault trials, 10 diagnosed/17.9%, 4 correct/40.0%, same per-fault-type and confusion-matrix breakdown) — confirming the tool is correct and that those specific numbers were not fabricated.

**Escalated finding — empty Retriever samples are far more severe than initially flagged.** The new script's sanity check found **180 of 189 possible Retriever samples in `logs/trials_new.jsonl` are empty strings (≈95%)** — i.e. the Retriever node is failing to produce usable output in nearly every sample across nearly every trial in this log, not just "some" trials as first noted. This means almost all Retriever-side lexical/Jaccard uncertainty values in the existing 66-trial log are measuring empty-vs-non-empty noise rather than genuine retrieval disagreement, and calls into question how much signal the `ceiling`/`contamination`-at-retriever detection numbers in Section 7 actually carry.

### 8.1 Root-Cause Investigation: Empty Retriever Samples

Traced through `src/nodes.py`'s `_ollama_generate()` (lines 71-99) and `src/config.py`'s token budgets. **Leading hypothesis, code-supported but not yet empirically confirmed with an isolated test call:**

- `OLLAMA_MODEL = "gpt-oss:20b-cloud"` (`config.py:48`) — OpenAI's open-weight `gpt-oss` series are **reasoning models**: by default they generate an internal chain-of-thought ("thinking"/analysis) pass before the final answer, and the Ollama API (client `ollama==0.6.2`, confirmed installed) returns that reasoning in a **separate `message["thinking"]` field**, distinct from `message["content"]`.
- `_ollama_generate()` (`nodes.py:96`) only ever reads `response["message"]["content"]`. It never inspects `message["thinking"]`.
- The response is post-processed with `re.sub(r"<think>...</think>", ...)` (`nodes.py:98`) — this is a **Qwen3-specific** cleanup (Qwen3 emits inline `<think>` tags in `content` itself) left over from before the branch switched backends (see `src/config.py`'s `MODEL = "mlx-community/Qwen3-8B-4bit"` at line 24, still the MLX-path default; `enable_thinking=False` is also only wired for the MLX path at `nodes.py:309`, not the Ollama path). **gpt-oss's reasoning never reaches this regex at all** because it isn't inlined in `content` the way Qwen3's is — it's already routed to the separate `thinking` field or, if `num_predict` is hit mid-reasoning, simply never gets to `content`.
- Retriever's token budget is `MAX_TOKENS = 128` (`config.py:56`), Writer's is the same; Reasoner gets `MAX_TOKENS_REASONER = 200` (`config.py:57`). These limits were calibrated for a non-reasoning Qwen3 call with thinking disabled. For a reasoning model with thinking *enabled by default* and no `enable_thinking=False`-equivalent set for the Ollama path, `num_predict=128` is very plausibly consumed entirely by internal reasoning tokens before the model starts emitting the actual `content` answer — which would produce exactly the observed pattern: `content` truncated to `""` in the large majority of calls, with Reasoner (more token budget: 200) still affected but perhaps less severely, and Writer/Retriever (128 each) hit hardest.

### 8.2 Confirmed and Fixed (same session)

Built `scripts/debug_gptoss_thinking.py` per the plan's Fix-2.4 isolated-test pattern and ran it directly against the Ollama backend, outside the full pipeline:

- **Test A** (production settings: `num_predict=128`, the Retriever's actual prompt): **5/5 calls returned `content_len=0`**, while `message["thinking"]` was consistently populated (519-573 chars). Confirms the hypothesis exactly — the reasoning pass alone exceeds a 128-token budget, so `content` is never reached.
- **Test B** (`think=False` passed to `ollama.chat`): reasoning length was **unchanged** (530-574 chars) and `content` was still empty in 5/5 calls — confirms `think=False` is **not honored** by the `gpt-oss:20b-cloud` proxy at `ollama==0.6.2`, ruling out fix option 1 from the prior write-up.
- **Test C** (`num_predict=1024`): `content` reliably appeared (~177-187 chars) alongside `thinking` (418-1049 chars). A follow-up sweep found **`num_predict=300` was already sufficient** for consistent non-empty Retriever content (3/3 calls, content_len≈177-179).

**Fix applied:**
- `src/config.py`: `MAX_TOKENS` raised `128 → 350` (Retriever/Writer), `MAX_TOKENS_REASONER` raised `200 → 600` (Reasoner needs headroom for both its own longer reasoning pass and a longer CoT+`FINAL ANSWER` output), both with margin above the measured minimums.
- `src/nodes.py`'s `_ollama_generate()` docstring updated to explain gpt-oss's separate `thinking` field, that `think=False` doesn't suppress it for this backend, and why `num_predict` must cover both passes.
- **Verified against the actual production function** (`_ollama_generate` at the new `MAX_TOKENS=350`, not just the standalone repro script): 5/5 calls returned non-empty content (`len=179` each).

### 8.3 Pilot Run Revealed the Token-Budget Fix Was Necessary but NOT Sufficient

A 2-question, k=5, 20-trial pilot (`logs/pilot_k5_fix.jsonl` + `logs/pilot_k5_fix_skip.jsonl`) was run against the new `MAX_TOKENS`/`MAX_TOKENS_REASONER` values to validate the fix empirically, per the plan's "run a pilot, manually inspect raw outputs" step (Section 5.5.4).

**Question 1** ("Were Scott Derrickson and Ed Wood...") worked cleanly across all 9 fault conditions + control: 0 empty Retriever samples out of 45, `FINAL ANSWER:` extraction clean, all three fault-target nodes (retriever/reasoner/writer) confirmed firing correctly.

**Question 2** ("What government position was held by the woman who portrayed Corliss Archer...") **completely broke the Reasoner and Writer nodes** — 45/45 Reasoner samples empty and 45/45 Writer samples empty (Writer depends on Reasoner's output, so it cascades) across all 9 fault trials for that question, despite the raised `MAX_TOKENS_REASONER=600`.

**Root cause of the residual failure**: an isolated test (same technique as `debug_gptoss_thinking.py`) showed gpt-oss:20b-cloud's reasoning length for this specific question is **non-deterministic and appears budget-seeking** — repeated calls at increasing `num_predict` produced `thinking` lengths of 1863, 4677, and 7622 chars respectively (roughly proportional to the available budget), with `content` still coming back **empty at num_predict=1200** despite being double the fixed value that worked for question 1. Neither `think=False` nor `think="low"` reliably suppressed this either (both still produced hundreds-to-thousands of chars of hidden reasoning). **A fixed token budget cannot fully solve this — the model will use however much reasoning budget it's given, unpredictably, for at least some questions.**

**Fix applied**: `src/nodes.py`'s `_ollama_generate()` now falls back to extracting an answer from `message["thinking"]` whenever `message["content"]` comes back empty — first searching `thinking` for a `FINAL ANSWER:` marker, then falling back to the last non-empty line of `thinking` as a best-effort answer. Verified this eliminates emptiness for the question-2 Reasoner case (0/5 empty in a follow-up isolated test, vs. 5/5 before).

**Residual data-quality caveat, not yet resolved**: the thinking-fallback text is **not equivalent in quality** to a natively-produced `FINAL ANSWER:` line. When `thinking` itself gets truncated by the token budget before the model reaches its own conclusion, the fallback grabs whatever the last reasoning line happens to be — observed examples from testing included incoherent fragments like `"Who"` (3 chars) and `"But maybe they want the specific government position..."` (mid-reasoning, not a conclusion). This is structurally the same class of problem the master plan warned about for `retroactive_extract_conclusion()` (Fix 2.2) — a heuristic grabbing near-marker text without confirming it's coherent — just triggered by truncation instead of a missing marker.

**Recommended follow-up (not yet implemented — flagged rather than done, to avoid silently expanding scope on this pass):** add a lightweight marker (e.g., a `used_thinking_fallback: true` field on the relevant sample/trial) whenever this fallback path fires, so downstream consumers (the verification script, any GNN training set) can filter or down-weight fallback-derived samples instead of treating them as equivalent to clean `FINAL ANSWER:` extractions. Until that exists, treat any trial with a very short or fragment-like sample as suspect and cross-check against `thinking` length before trusting it.

**Not yet done as further follow-ups:**
- The existing 66-trial `logs/trials_new.jsonl` was generated entirely under the old broken `MAX_TOKENS=128`, so its Retriever-side uncertainty values remain unreliable and should not be used for the Section 7 numbers going forward — the new pilot (`logs/pilot_k5_fix.jsonl`) is a better (though still small) reference, and a larger fresh run is still needed.
- Token budgets and the fallback logic have only been validated against 2 example questions; question-to-question variance in required reasoning length appears large, so more spot-checking is warranted before a full-scale run.
- The pilot's own diagnostic accuracy was weak (see numbers below) — separate from the data-quality fix, worth investigating once the fallback-flagging follow-up above lets bad samples be excluded from that analysis.

**Pilot verify_results.py output** (for reference — small sample, not conclusive; includes the still-degraded question-2 trials):
```
Fault trials: 17, Diagnosed: 11 (64.7%), Correct among diagnosed: 3/11 (27.3%)
  ceiling:        4/5 detected (80.0%), 3/5 correct
  contamination:  3/6 detected (50.0%), 0/6 correct
  noise:          4/6 detected (66.7%), 0/6 correct
```
Detection rate went up sharply (64.7% vs. the old 17.9%) but accuracy among diagnosed dropped (27.3% vs. 40.0%) — plausibly because question 2's degraded reasoning/writer samples are injecting spurious "uncertainty" unrelated to the actual injected fault, inflating detections while corrupting correctness. This is exactly the kind of contaminated-signal risk the fallback-flagging follow-up above is meant to address.

---

## 9. Follow-Up Round (external review) — Six Additional Verification Items

An external review of this document (same date) flagged six gaps between what the doc claimed and what had actually been executed/verified. All six were addressed:

### 9.1 Compute/cost estimates are stale — real measurement (item 1, "most consequential")

The review correctly flagged that raising `MAX_TOKENS`/`MAX_TOKENS_REASONER` (Section 8.2) plus raising `k` 3→5 (Section 8) invalidates every earlier compute/cost estimate in this project's history, which assumed the old 128/200 budgets, k=3, and no hidden-reasoning pass at all.

**Measured directly** (via `ollama.chat`'s real `eval_count`/`prompt_eval_count` fields, not character-length guesses) across 12 isolated calls on 2 example questions:

| | avg output tokens/call | avg latency/call |
|---|---|---|
| Retriever, easy question | ~112 | ~2.7s |
| Retriever, hard question | ~339 (at/near the 350 cap) | ~5.4s |
| Reasoner, easy question | ~173 | ~3.7s |
| Reasoner, hard question | ~497 (at/near the 600 cap) | ~8.0s |

**Key finding: harder questions don't just use more tokens than easy ones — they pin the token budget itself** (`eval_count` hit exactly `MAX_TOKENS`/`MAX_TOKENS_REASONER` on 2/3 calls for the hard question at both nodes). This means per-call cost is not a fixed number; it scales with question difficulty up to whatever cap is set, and a meaningful fraction of "hard" trials will cost the full budget every time.

**Combined multiplier vs. the original master-plan estimate** (which assumed k=3, MAX_TOKENS=128, MAX_TOKENS_REASONER=200, and implicitly no hidden-reasoning overhead): `k` alone is 5/3 ≈ 1.67x more calls per node; per-call token budgets are ~2.7-3x higher. **Combined, expect roughly 4.5-5x more total output tokens generated for the same trial count than any prior estimate accounted for.** Any Kaggle GPU-hour budget, Ollama Cloud usage-tier assumption, or wall-clock estimate calculated before this session should be treated as invalid and re-measured at the new settings before committing to a ~400-trial run. Latency-wise, the one real pilot run (20 trials, full grid, 2 questions) took ~30-45 minutes wall-clock under the *already-updated* budgets — scaling this to ~400 trials gives a rough order-of-magnitude of several hours to a day, but this has real uncertainty (driven by how many trials are "hard" and hit the cap, and how often diagnosis triggers retry batches) and should be re-checked at a larger pilot size (Section 10.6) rather than trusted as-is.

**Not yet done**: actual Ollama Cloud usage-quota consumption was not checked against a real account tier (no access to that from this session) — the user should verify this directly against whichever tier they're on before scaling up, per the original master plan's Section 5.1 guidance.

### 9.2 NODE_THRESHOLDS not recalibrated for k=5 — still open

Confirmed still true, no change since Section 8. This remains the single most important open item before trusting any k=5 detection numbers — the achievable uncertainty value set changed shape (k=3: `{0, 0.333, 0.667}`; k=5: `{0, 0.2, 0.4, 0.6, 0.8, 1.0}`) and `NODE_THRESHOLDS` (`config.py:75-81`) were never re-verified against fresh k=5 clean-baseline trials.

### 9.3 `used_thinking_fallback` flag — implemented (upgraded from "recommended" to required)

Per-sample tracking now flows through the full pipeline:
- `src/nodes.py`: `_ollama_generate()` returns `(text, used_fallback)`; `_generate_once()` propagates it (always `False` for mock/MLX); `sample_node()` collects a `used_thinking_fallback: List[bool]` per node, same length as `samples`, stored on `NodeResult`.
- `scripts/run_study1.py` and `scripts/run_study2.py`: trial records now include a `used_thinking_fallback` dict (only for nodes where at least one sample used it, to keep clean trials' records unchanged).
- `scripts/verify_results.py`: new `--exclude-fallback` flag reports what fraction of trials used the fallback path and can re-run all detection/accuracy numbers with those trials excluded, to directly test the review's hypothesis that fallback-contamination explains the pilot's accuracy drop (40.0% → 27.3%).

**Verified working** via a mock-mode run (`run_study1.py --mock`) confirming the field serializes correctly and `verify_results.py --exclude-fallback` runs without error (0% fallback rate in mock mode, as expected — mock never triggers the Ollama-specific fallback path). **Not yet re-tested against a real backend run** — the existing `logs/pilot_k5_fix.jsonl` predates this change and has no `used_thinking_fallback` data to filter on; a fresh pilot is needed to get real before/after accuracy numbers (see Section 10.6).

### 9.3a `logs/pilot_k5_fix.jsonl` is triple-confounded, not just "small and old"

Section 9.4 below found `build_prompt()` had been silently dropping the old chain's trailing continuation cues and field order for every run up to and including this session — which means `logs/pilot_k5_fix.jsonl` (referenced in Section 8.3 as "a better, though still small, reference") was generated **before that prompt fix too**. Its 64.7%/27.3% numbers are therefore confounded by three separate issues stacked on top of each other, not one: (1) partial token-budget effects from question-to-question variance, (2) no `used_thinking_fallback` flag existing yet to filter contaminated samples, and (3) a pipeline missing the exact continuation cues that plausibly help the model produce cleanly-formatted output in the first place — meaning the format instability chased in Section 8.3 may have had a second, previously-invisible contributor beyond simple token truncation. **`logs/pilot_k5_fix.jsonl` should not be treated as a reference for anything past Section 8's narrow purpose (confirming the empty-sample fix worked for question 1)** — a new pilot combining every fix is needed before any number from this project is trustworthy (Section 9.7).

### 9.4 Two "done" claims needed isolated-test rigor — one passed clean, one found a real bug

**Noise-temperature scoping (Fix 2.4)**: built `scripts/debug_noise_scoping.py`, which runs the real `run_pipeline()` (mock backend) with a monkeypatched `sample_node` spy, capturing the actual temperature passed to every node's generation call across all three possible noise targets. **Result: PASS on all 3 targets** — only the targeted node ever receives `NOISE_TEMPERATURE`; the other two always receive `DEFAULT_TEMPERATURE`. This claim now has real executed-test evidence, not just code inspection.

**Topology-engine regression test (plan Section 4.1)**: built `scripts/debug_topology_regression.py`, diffing the new generic `build_prompt()`'s output against the old hardcoded `retriever_prompt()`/`reasoner_prompt()`/`writer_prompt()` functions on 2 test questions. **This FAILED on first run** — the generic engine had silently dropped the trailing continuation cue (`"Relevant sentences:"` / `"Reasoning:"` / `"Final answer:"`) that every old prompt ended with, and had also swapped field order (Paragraphs-before-Question instead of Question-before-Paragraphs). This is a genuine, previously undetected regression — not cosmetic, since a missing continuation cue plausibly contributes to the exact reasoning/format instability chased in Section 8 (no explicit "continue here" signal for the model). **Fixed** in `src/pipeline.py`'s `build_prompt()`: restored the role-specific trailing cue and Question-first field order for the single-parent case (matching the old chain exactly), while keeping the generic `[Input from <parent_id>]:` labeling for genuinely multi-parent nodes (e.g. a dual-retriever fan-in) where no single old-style label applies. Re-ran the regression test after the fix: **all 6 checks (3 nodes × 2 questions) now PASS, byte-for-byte identical to the old hardcoded prompts.** Also re-ran the mock full pipeline and a mock `run_study1.py` end-to-end after the fix — both run clean with sane uncertainty values, no exceptions.

### 9.5 Jaccard uncertainty unit test — added and passing

`tests/test_uncertainty.py` now exists with 5 tests, including the plan's exact worked example. Result: `jaccard_uncertainty({a,b},{a,b,c},{a,d},{a,b,d})` = **0.4861**, matching the plan's expected ~0.49. All 5 tests pass (identical-sets → 0.0, disjoint-sets → 1.0, single-sample → 0.0, `per_item_inclusion_frequency` values all correct).

### 9.6 `--no-diagnose` flag added — cuts the diagnosis-retry cost for bulk generation

The user was asked directly whether `diagnose_trace()` (the retry-then-reprobe protocol) should stay active for the full ~400-trial run, since it triggers up to 3 extra k-sample batches per flagged node (2 same-input retries + 1 clean-input retry by default) — a meaningful chunk of the Section 9.1 cost multiplier. **Decision: disable it for the bulk run** — the dataset's purpose is raw per-node uncertainty signals + `true_label` as training features, not evaluating the retry-heuristic's own diagnostic accuracy.

Implemented as `--no-diagnose` on both `scripts/run_study1.py` and `scripts/run_study2.py`: skips `diagnose_trace()` entirely for every fault trial while still recording all raw uncertainties/samples/`used_thinking_fallback` flags. `diagnosed_label`/`per_node_diagnoses` are left at their `PipelineTrace` defaults (`None` / `{}`) — distinct from `"no_fault_detected"`, which means diagnosis ran and found nothing. Confusion-matrix building is skipped with an explanatory message rather than silently producing an empty/misleading matrix. Verified via mock-mode runs on both scripts — correct behavior confirmed (`diagnosed_label: null`, `per_node_diagnoses: {}`, all other fields populated normally).

### 9.8 Real combined-fix pilot — COMPLETE

`logs/pilot_combined_v2.jsonl` (2 questions, k=5, full 10-condition grid = 20 trials — 19 trials + 1 skip, real Ollama backend, diagnosis active since this run started before `--no-diagnose` existed) validated every fix from this session together for the first time: the corrected `build_prompt()` (Section 9.4), the `used_thinking_fallback` flag (Section 9.3), k=5, and the raised token budgets (Section 8.2). Final `verify_results.py` output:

```
Trials using thinking-fallback (any node/sample): 10/19 (52.6%)
  Fallback sample word counts: min=1 median=47 max=463 (4/83 are <=3 words -- likely unusable fragments)

Fault trials: 17
Diagnosed (non-null): 17 (100.0%)
Correct among diagnosed: 6/17 (35.3%)
  ceiling:        5/5 detected (100.0%), 5/5 correct
  contamination:  6/6 detected (100.0%), 1/6 correct
  noise:          6/6 detected (100.0%), 0/6 correct
Confusion matrix: ('ceiling','ceiling'):5  ('contamination','ceiling'):5  ('contamination','contamination'):1  ('noise','ceiling'):6
Empty retriever samples: 0
```

**What this confirms (good news):** the empty-sample bug is fully resolved on this combined run — **0 empty Retriever samples**, vs. ~95% in the original 66-trial log. The `build_prompt()`/fallback/k=5 fixes work together correctly against the real backend, not just in mock mode.

**What this reveals (new finding, not previously visible):** the thinking-fallback rate is much higher than the earlier single-question spot checks suggested — **52.6% of trials needed it**, not an occasional edge case. The fallback text itself is mostly usable (median 47 words), but a real minority (4/83 samples, ~4.8%) are unusable fragments (≤3 words) — confirming the Section 8.3 concern was correctly calibrated, not overstated.

**What this reveals (a second, more severe finding): the diagnostic protocol is badly miscalibrated at k=5 — everything gets diagnosed as "ceiling."** Noise: 0/6 correct (100% misdiagnosed as ceiling). Contamination: 1/6 correct (5/6 misdiagnosed as ceiling). Only ceiling itself is correctly diagnosed (5/5). This is a strong, concrete confirmation of Section 9.2's still-open item — `NODE_THRESHOLDS` were never recalibrated for k=5 — and this pilot shows the consequence isn't cosmetic: the retry-then-reprobe protocol is currently non-functional as a 3-way classifier, collapsing to "always guess ceiling." **This does not block the current path forward** (Section 9's later decision to disable diagnosis via `--no-diagnose` for bulk generation, and to switch to Qwen2.5-7B-Instruct for generation entirely — see `plan.md`), but it means gpt-oss's retry-then-reprobe accuracy numbers from any session-era log (including this one, and `pilot_k5_fix.jsonl`) should not be read as "the diagnostic system doesn't work" in general — they reflect an uncalibrated threshold, not a fundamental flaw in the retry-then-reprobe design. This should be re-examined if/when the retry-then-reprobe protocol is later evaluated as the naive-baseline comparator against belief propagation / GNN methods (per the original research plan) — at that point it'll need proper threshold recalibration, on whichever model is used then.

**On `pilot_k5_fix.jsonl`'s accuracy drop (Section 9.3a)**: this run doesn't cleanly isolate that question, since this pilot itself has both the fallback confound (52.6% rate) and the now-revealed threshold-miscalibration issue layered together. Not a clean before/after comparison — treat Section 9.3a's confound concern as still valid, just not further disentangled by this run.

### 9.7 Topology pool (train + OOD) — DONE (later same session)

Was the largest remaining item at the time this section was first written. **Since built**: `dataset/generate_topology_pool.py` (in the top-level `dataset/` folder). See Section 10.1 for full details — 14 train + 6 OOD topologies, frozen to `dataset/topology_pool.json`, validated and deduplicated.

---

## 10. Model Switch to Qwen2.5-7B-Instruct-AWQ + Real Calibration Results

Per `plan.md`, generation has moved off gpt-oss/Ollama entirely to **Qwen2.5-7B-Instruct-AWQ served via vLLM** on Kaggle GPU (T4 x2), driven by vLLM's real batching throughput and Qwen2.5 having no hidden-reasoning-pass problem class at all. This section covers what's been built and verified for that switch.

### 10.1 Infrastructure built this session

- **`BACKEND='vllm'` added to the pipeline** (`src/config.py`, `src/nodes.py`): `_get_vllm_llm()` (lazy singleton loader) and `_vllm_generate()`, dispatched from `_generate_once()` alongside the existing `ollama`/`mlx`/mock paths. Reuses the exact same `run_pipeline()`/`sample_node()`/uncertainty code — not a parallel implementation — so whatever gets calibrated here is what bulk generation will actually run through.
- **`scripts/calibrate_vllm_model.py`** — runs real example questions through the pipeline (clean control only) and reports token usage, `FINAL ANSWER:` compliance, a `parse_retrieved_items` audit, achievable uncertainty spread, and suggested `NODE_THRESHOLDS`.
- **Kaggle CLI verified working end-to-end from this machine**: authenticated via OAuth already, GPU quota confirmed (29.99h/30h remaining), a plain GPU smoke test confirmed 2x Tesla T4 (15360MiB each) with a real CUDA matmul executing successfully.
- **`dataset/generate_topology_pool.py`** (in the top-level `dataset/` folder, sibling to `btp-pipeline/`) — generates and freezes the train (14 topologies: chain, fan-in/fan-out variants, deep chains, star, tree, wide fan-in/out, asymmetric) + OOD (6 topologies, random-DAG MOC-style construction, role-before-edges so valid by construction) pool to `dataset/topology_pool.json`. Deduplicated, round-trip-loadable, all validated via `validate_topology()`. This is pure Python, no GPU needed, and is done.
- **A real, previously-live bug found and fixed via this work**: a mock dry-run of the calibration script caught `parse_retrieved_items` still shredding a normal sentence with internal commas (e.g. `"...director, screenwriter, and producer."` → 3 fragments) via a leftover comma-fallback branch — the same failure class the original numeric-comma fix targeted, just triggered differently. Removed the comma fallback entirely in `src/uncertainty.py`; newline-only splitting is correct for the Retriever's "one sentence per line" prompt format. Verified fixed both via direct testing and confirmed clean in the real Kaggle calibration run below (no more fragment-shredding observed).

### 10.2 Real calibration run — Kaggle GPU, Qwen2.5-7B-Instruct-AWQ, 2026-09-24

3 example questions, k=5, clean control only (45 total samples across 3 roles). Required 3 iterations to get the Kaggle kernel plumbing right (dataset mount path was actually `/kaggle/input/datasets/<user>/<dataset>`, not `/kaggle/input/<dataset>` as first assumed; a `scripts/` package-import issue). Once fixed, ran cleanly in ~5 minutes including vLLM install and model load (~100s engine init).

**Results:**
- **0/45 empty samples across all roles** — confirms Qwen2.5 has no reasoning-model truncation problem; `MAX_TOKENS=350`/`MAX_TOKENS_REASONER=600` (unchanged, gpt-oss-era values) are already comfortably sufficient (observed max: retriever 94 words, reasoner 226 words, writer 31 words).
- **`FINAL ANSWER:` marker compliance: 15/15 (100%)** on Reasoner samples.
- **`parse_retrieved_items` audit (the manual check the original master plan required but was never previously done): clean.** No fragment-shredding observed on any of the 15 real Retriever samples — the comma-fallback removal (Section 10.1) holds up on real data.
- **New finding, not a bug**: the Retriever is not filtering to "only relevant sentences" as its prompt instructs — it mostly echoes back full paragraphs verbatim (including the `[Title]` bracket formatting from the input context block). E.g. for the Scott Derrickson/Ed Wood question, it returned both full paragraphs rather than the single relevant sentence from each. `parse_retrieved_items` handles this correctly (splits into paragraph-level items via newlines, no fragmentation) — this is an instruction-following gap in the model's behavior, not a parsing bug. Worth a prompt tweak (e.g. explicitly caution against copying full paragraphs) if tighter Retriever filtering matters for the dataset's quality; left as-is for now since it isn't broken, just more verbose than originally designed for.
- **Uncertainty values, small-sample caveat**: Reasoner's semantic uncertainty swung 0.0 → 1.0 → 0.42 across the 3 questions (mean 0.47, std 0.41) — real per-question variance, not yet a stable calibrated number. Writer: mean 0.14, std 0.20. `NODE_THRESHOLDS` in `config.py` were **not** updated from these — still gpt-oss-era values, explicitly left alone since (a) diagnosis is disabled for generation so they don't matter yet, and (b) n=3 questions is too small a sample to treat as real calibration. Documented in `config.py`'s comments for whenever the retry-then-reprobe protocol is evaluated later as the naive-baseline comparator.

### 10.3 `scripts/run_study_vllm.py` — written, logic-verified locally, GPU call not yet smoke-tested

The batched generation script (per `plan.md` Section 4.3) now exists. Design:

- **Reuses, does not reimplement**: `build_prompt`, `apply_fault_to_prompt`, fault injectors (`inject_noise`/`inject_contamination_retriever`/`inject_contamination_downstream`/`inject_ceiling`), `verify_against_baseline`, and `_trace_to_record`/`_skip_record`/`build_substitute_bank` (imported directly from `run_study2.py`). The only new logic is the batched per-node execution loop itself.
- **A refactor was needed first, to avoid a second uncertainty-computation implementation**: `src/nodes.py`'s `sample_node()` had its post-generation uncertainty math (lexical/semantic/Jaccard) factored out into a new standalone `compute_node_result_from_samples()`. The batched path calls this exact same function after getting k samples back from one vLLM call per node (via `SamplingParams(n=k)`), instead of `sample_node`'s sequential k-loop. This was done specifically to avoid repeating the class of bug that caused the `build_prompt` continuation-cue regression (Section 9.4) — two divergent implementations of "the same computation."
- **Two-phase batching per topology**: Phase 1 batches every question's control trial together to establish per-question baselines; Phase 2 resolves every fault trial's injection (skip-and-log invalid ones, identical logic to `run_study2.py`) *before* generation, then batches all resolved fault trials together, verifying each against its own question's Phase 1 baseline.
- **Diagnosis is never called** (no `diagnose_trace` import at all) — consistent with the `--no-diagnose` decision; raw uncertainties + verified `true_label` are recorded, `diagnosed_label` stays `None`.
- **A pre-existing gap found while reading `run_study2.py`'s fault logic (not introduced by this work, not fixed here)**: `_hardened_instruction` (meant to harden a downstream node's own instruction for a ceiling fault) is set into the fault-config dict in both `run_study1.py` and `run_study2.py` but is never actually read anywhere in `pipeline.py` — dead data. `run_study_vllm.py` faithfully replicates this existing (already-present) behavior rather than silently fixing it; flagged in the new script's comments for a future pass.

**Verified locally, without a GPU, via `--mock`:**
- Full dry run (2 topologies, 2 examples, mock backend) completes cleanly end-to-end, produces correctly-shaped records, `verify_results.py` runs against the output without error.
- **Batched fault-config isolation directly tested**: a monkeypatch spy on `apply_fault_to_prompt` across a 3-trial mixed batch (`noise@retriever`, clean, `noise@writer`) confirmed it fires exactly twice — once for each correctly-targeted (trial, node) pair, never for the clean trial or the wrong node. This is the plan.md Section 5.5 item 3 check, passing at the orchestration-logic level.
- Full regression suite (Jaccard unit tests, `debug_noise_scoping.py`, `debug_topology_regression.py`) re-run clean after the `sample_node` refactor — no regressions introduced.

### 10.4 Kaggle GPU smoke test — PASSED, 2026-09-24

Ran `scripts/run_study_vllm.py` for real against Qwen2.5-7B-Instruct-AWQ on Kaggle GPU (2 topologies — `chain`, `dual_retriever_fanin` — 2 examples, k=3, `--batch-size 15`). All `plan.md` Section 5.5 checks passed:

- **GPU confirmed in use** — real weight loading (5.29 GiB), CUDA graph capture, GPU temp rose 44°C → 73°C across the run (not a silent CPU fallback).
- **`llm.chat()`'s batched form confirmed working** — the one previously-unverified assumption in the new code (`List[List[message]]` + matching `List[SamplingParams]` in one call). No errors, correct per-trial outputs returned.
- **`--batch-size 15` worked with no OOM** — chain topology's 18 fault trials ran as 2 sub-batches (28.7s + ~11s), `dual_retriever_fanin`'s similarly (~59s total for its full grid).
- **Exit code 0, 41/41 trial records written** — exactly matching the mock dry-run's record count (19 + 22), confirming the real run's control flow matches the logic-verified mock path.
- `verify_results.py` run against the real output: **0/41 empty samples, 0% thinking-fallback rate** (as expected — Qwen2.5 has no reasoning-model fallback path), skip pattern (5 `ceiling_injection_failed_*`) identical to the mock dry-run's skip pattern. `diagnosed_label` correctly `null` throughout (diagnosis intentionally never invoked).
- The Kaggle code dataset (`btp-pipeline-code`) was updated to a new version including `run_study_vllm.py`, the refactored `src/nodes.py`, and `dataset/topology_pool.json` before this run — previously flagged as needed, now done.

**Batched fault-config isolation** (Section 5.5 item 3) was verified at the orchestration-logic level in Section 10.3's mock-mode spy test, not independently re-verified against real generated text in this GPU run (harder to check directly from output alone) — the logic-level test is considered sufficient since the isolation happens before generation (in prompt/temperature assembly), identically regardless of backend.

**Everything in `plan.md`'s Definition of Done (Section 7) is now checked off** except the full-scale generation run itself and its post-hoc `verify_results.py` check — the pipeline is generation-ready.

## 11. Bottom Line

The `corrected-pipeline` branch is substantially further along than any prior written summary (including the previous version of this file) credited it for. The topology-generalization and fault-taxonomy redesign — originally scoped as a large, mostly-unstarted body of work — is in fact implemented and structurally correct. `k` is now 5 and a reusable `verify_results.py` exists, closing both previously-open gaps.

The empty-sample data-quality bug is **substantially, but not completely, fixed**. The token-budget increase (Section 8.2) eliminated the failure mode seen in the original 66-trial log for one pilot question; a second pilot question exposed a deeper issue — gpt-oss:20b-cloud's reasoning length is non-deterministic and budget-seeking, so no fixed `num_predict` value is guaranteed safe for every question. The thinking-field fallback (Section 8.3) now guarantees non-empty samples in all cases, but at the cost of sometimes returning a low-quality, truncated-reasoning fragment instead of a clean answer. **This is no longer untracked**: Section 9.3's `used_thinking_fallback` flag now marks every affected sample, and `verify_results.py --exclude-fallback` can isolate clean-path-only numbers from fallback-contaminated ones.

The Section 9 follow-up round (external review) closed five of six items and surfaced one genuine new bug: the topology engine's `build_prompt()` had silently diverged from the old hardcoded chain (missing continuation cues, swapped field order) despite the plan explicitly requiring a regression test to catch exactly this — now fixed and verified byte-identical. Noise-temperature scoping and Jaccard uncertainty both now have real executed-test evidence rather than resting on code-reading alone. The token-budget fix's true cost was also quantified directly (not guessed): expect **roughly 4.5-5x more total output tokens** for the same trial count than any pre-this-session estimate assumed, driven by both the k 3→5 increase and harder questions reliably pinning the raised token budgets.

**Net effect on trust in numbers so far:** none of the detection/accuracy percentages produced to date (the original 17.9%/40.0%, or the pilot's 64.7%/27.3%) should be treated as representative of the system's true performance — all were generated under a gpt-oss/Ollama pipeline now superseded for generation purposes (Section 10). **Remaining before full-scale generation** (updated, see Section 10.3 for the current concrete list): write `scripts/run_study_vllm.py` (the batched generation script) and smoke-test it on Kaggle. The topology pool (Section 9.7) and Qwen2.5 calibration (Section 10.2) are both done. `NODE_THRESHOLDS` recalibration remains open but is not a generation blocker since diagnosis is disabled for the bulk run.
