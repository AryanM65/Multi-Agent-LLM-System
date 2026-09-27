# Future Work — After the First Trained Model

> This is the "what's next" list, written 2026-09-26, once the first GNN (per `model-plan.md`) is trained and evaluated. None of this blocks getting a first working model — it's the roadmap for making the thesis stronger after that baseline result exists. Grouped by theme, roughly in priority order within each group.

---

## 1. Multiple faults at once (currently: only single-node faults exist)

**Current limitation**: every trial in `dataset/trials.jsonl` has **at most one faulted node** (`fault_config = {"type": ..., "target_node": <one node>}`, or none for controls). The model we're building only ever has to answer "which single node is broken." Real multi-agent pipelines can fail in more than one place at once, and a model trained only on single-fault examples has no reason to handle that case correctly.

**What's needed**:
- **Generate a new batch of "multi-fault" trials**: 2 (and maybe 3) nodes faulted simultaneously in the same run, with independent fault types per node (e.g. `noise` at Retriever *and* `contamination` at Writer in the same trial). Requires extending `src/faults.py`'s injection functions to accept a *set* of `(node, fault_type)` pairs rather than a single one, and extending `fault_config` to a list.
- **Decide the label representation**: single-fault used a one-hot vector over nodes; multi-fault needs a **multi-label** target (a 0/1 vector where more than one entry can be 1) instead of a single-class softmax. This changes the loss function from cross-entropy to something like binary cross-entropy per node.
- **Decide how "severity" combines**: if two faults are injected on adjacent nodes, does the downstream node's uncertainty reflect fault #1, fault #2, or a blend? Needs its own verification step (same rigor as the single-fault `verification.z_score` check) so multi-fault trials are still *verified* ground truth, not just "we intended two faults so we're trusting it worked."
- **New model, or extend the current one**: the same GNN architecture likely still works (swap the output head from single-class softmax to per-node binary/sigmoid), but this needs its own baseline comparison — a model that's great at localizing exactly-one-fault might do much worse (or, encouragingly, might do fine) on two-fault graphs. Report separately, don't blend into the single-fault numbers.
- **Interesting sub-question for the thesis**: can the model tell "one fault whose symptom spread to two nodes" apart from "two independent faults"? This is exactly the kind of question a naive threshold rule cannot answer but a GNN's structural reasoning plausibly could — a good differentiator to highlight if it works.

---

## 2. Richer uncertainty signals (currently: 3 uncertainty metrics, some role-restricted)

**Current limitation**: only `lexical_uncertainty` (always), `semantic_uncertainty` (Reasoner/Writer only), and `jaccard_uncertainty` (Retriever, multi-item only) are computed. Several richer signals already exist in the underlying code but aren't wired into the dataset:

- ~~**`item_frequencies` / `per_item_inclusion_frequency`**~~ — **done, 2026-09-27** via `model/data/enrich_features.py`'s `compute_item_frequencies()`, a post-hoc pass (no regeneration needed). Populated for 1088/1762 trials (the rest are non-multi-item Retriever outputs, correctly left `None`). Not yet isolated in its own ablation — was added alongside `inference_gaps` and `node_embeddings` in one combined run.
- ~~**`inference_gaps`**~~ — **done, 2026-09-27** via `model/data/enrich_features.py`'s `compute_inference_gaps()`, also a post-hoc pass reusing the already-stored per-node sample texts (approximates node input as the parent's own sample text + question, rather than the exact original prompt — a reasonable stand-in, not identical to `diagnose_trace`'s definition). Populated for all 1762 trials. The dedicated "does this help `ceiling`-fault localization specifically" experiment mentioned here was not done separately — it went in bundled with the other enrichment fields.
- **Node embeddings** (not originally listed here, but same theme): raw sentence-embeddings per node, added and found to require PCA compression + input-layer regularization to actually help OOD (see `docs/model/model.md` Section 8) — the single biggest lever tried this round.
- **Calibration / confidence recalibration**: `NODE_THRESHOLDS` (used by the never-invoked `diagnose_trace` retry-then-reprobe baseline) were never recalibrated for Qwen2.5's own baseline behavior (a stale open item carried over from earlier calibration work). If `diagnose_trace` is later used as a comparison baseline (Section 4 below), this recalibration needs to happen first, or the comparison is unfair (comparing the GNN against a badly-tuned heuristic).
- **Cross-node correlation features**: currently every node's features are computed independently; a graph-level feature like "variance of uncertainty across all nodes in this trial" or "rank of this node's uncertainty among its neighbors" isn't provided directly — the GNN is expected to learn this via message passing, but an explicit engineered version could be a useful ablation to check whether message passing is actually adding value over hand-crafted relative features.

---

## 3. Faults at multiple *depths* / more complex fault interactions

- **Cascading fault sensitivity**: does a fault at node A in a *deep* chain (many hops before the symptom is visible) get localized as accurately as the same fault type at a node right before the final output? The current dataset has topologies of varying depth (`deep_chain`, `deep_chain_5node`), so this analysis is possible with existing data — it's an evaluation/analysis task, not a new-data task. Do this before generating any new data, since it might already show whether the model's failure mode is depth-related.
- **Fault-type combinations beyond the 3 existing ones**: `noise`, `contamination`, `ceiling` are the only 3 types. Real pipeline failures could include things like: a node timing out and returning a truncated/empty response, a node hallucinating a plausible-sounding but entirely fabricated fact (distinct from `contamination`, which swaps in a *real* wrong answer from elsewhere), or a prompt-injection-style fault where a node's instructions are silently overridden. Each would need its own injection function in `src/faults.py`, its own verification method, and its own labeled examples.

---

## 4. Comparison baselines beyond the naive max-uncertainty rule

- **`diagnose_trace`'s retry-then-reprobe heuristic** (`src/diagnose.py`) was deliberately never run during dataset generation (kept the dataset "clean" for training a better method), but running it as a **separate post-hoc pass** over the existing trials would give a second, non-trivial baseline to compare the GNN against — closer to "the best simple heuristic we could build" rather than just "the dumbest possible rule." Needs the `NODE_THRESHOLDS` recalibration from Section 2 first.
- **Non-graph ML baseline** (flat feature table + XGBoost/logistic regression, mentioned in `model-plan.md` Step 4 as optional) — worth actually building once the GNN is done, specifically to answer "is the graph structure/message-passing doing real work, or would a much simpler tabular model do just as well?" This is a standard and expected ablation for any GNN-based thesis result.
- **Belief propagation** (a classical, non-learned alternative to a learned GNN) — a classical, non-learned message-passing algorithm. Comparing it against the learned GNN answers "do we need to *learn* the propagation rule, or does a hand-designed one already work about as well?" — a meaningful comparison point for the thesis's contribution claim.

---

## 5. Expanding the dataset itself

**Current dataset**: 493 trials, 20 topologies (14 train / 6 OOD), 30 questions, single model (Qwen2.5-7B-Instruct-AWQ), single fault per trial, k=5 samples.

Possible expansions, roughly by expected effort/impact:

- **More questions** (currently only 30, all HotpotQA-distractor). More question diversity would reduce the risk of the model latching onto question-specific quirks rather than genuine uncertainty-propagation patterns. Cheapest expansion — same generation pipeline, just point `sample_examples.py` at a larger pool.
- **More topologies**, especially more OOD topologies (currently only 6) — a bigger, more diverse OOD test set gives a more reliable generalization estimate (6 topologies is a small sample to draw strong conclusions from). `generate_topology_pool.py` already supports random-DAG generation; just needs a larger run.
- **A second base model** (e.g. a different open-weight model than Qwen2.5, or a larger/smaller variant) — tests whether the uncertainty-propagation patterns the GNN learns are Qwen2.5-specific artifacts or genuinely general across models. Important for the thesis's generality claim; also the most expensive item on this list (needs a full new Kaggle generation run, likely another `NODE_THRESHOLDS`/`MAX_TOKENS` recalibration pass, same process used when the model backend was first switched).
- **Higher k** (more than 5 self-consistency samples per node) — would give smoother, more continuous uncertainty values instead of the current 6-level discrete set `{0, 0.2, 0.4, 0.6, 0.8, 1.0}` at k=5 (`dataset_description.md` §5). Costs proportionally more Kaggle GPU time per trial; a targeted small-scale test (higher k on a handful of topologies) could first confirm whether discreteness is actually limiting the model before committing to a full higher-k regeneration.
- **Filling the `retriever_c` coverage gap** (documented in `btp-pipeline/model.md` Section 3, item 5) — currently accepted as a known limitation for the first model, but if the model's evaluation shows this gap is actually hurting generalization (e.g. the model is systematically bad at handling "third parallel retriever" nodes in the OOD topologies because it never saw an analogous training example), a small targeted regeneration (just the missing fault conditions for those 2 topologies) would be cheap to run.
- **Full multi-parent contamination fix** (documented in `btp-pipeline/model.md` Section 3, item 4) — currently only the first parent of a multi-parent node gets corrupted per contamination trial. Fixing this in `src/faults.py` and regenerating just the affected trials would sharpen the contamination-fault signal at fan-in nodes specifically.

---

## 6. Model architecture / training improvements

- **Try GraphSAGE / GCN as alternatives to GAT**, and compare — `model-plan.md` recommends starting with GAT but flags this as worth testing, not a settled choice.
- **Bidirectional message passing** — test whether adding reverse edges (so information can flow both forward, matching real pipeline execution, and backward, matching how a fault symptom needs to be traced back to its cause) improves localization accuracy over forward-only edges. This is flagged as an open, untested decision in `btp-pipeline/model.md` Section 1 and should be one of the first ablations run once the base GNN works.
- **Multi-task learning** — jointly predicting fault-type (`noise`/`contamination`/`ceiling`/`clean`) alongside node localization, via a graph-level auxiliary head (mentioned as optional in `model-plan.md`). Worth testing whether this regularizes the node-localization task or actually hurts it (multi-task learning doesn't always help — needs an honest ablation, not an assumption).
- **k-fold cross-validation within the train split** — given the small dataset size (only 329 train-topology trials), a single train/val split's numbers carry real variance. Report results across multiple folds/seeds rather than one run, per `model-plan.md`'s "practical notes."
- **Interpretability pass**: since GAT layers produce attention weights (how much each node weighted each neighbor), inspect these for a handful of correctly- and incorrectly-localized trials — do the attention weights "point at" the true fault node in a way a human could sanity-check? This is a strong qualitative result for a thesis write-up if it holds up, and a revealing failure-mode diagnostic if it doesn't.

---

## 7. Evaluation depth

- **Statistical significance testing** between the GNN and each baseline (naive rule, `diagnose_trace`, non-graph ML) — with under 500 examples split three ways, a few-percentage-point accuracy difference could easily be noise. A paired test (e.g. McNemar's test on the OOD set, since it's the same set of examples scored by multiple models) would make any claimed improvement defensible.
- **Per-topology OOD breakdown**, not just pooled OOD accuracy — with only 6 OOD topologies (24-32 trials each), report all 6 individually; a pooled number can hide "does great on 5, terrible on 1."
- **Confidence vs. correctness calibration** — does the model's predicted-probability for its top guess actually track how often it's right? (E.g., among trials where the model was 90%+ confident, is it right ~90% of the time?) Useful both as a sanity check and as a potential future signal (a low-confidence prediction could trigger "ask an oracle" in a real deployed system — ties back to the original motivating question of avoiding an external oracle, worth a paragraph in the thesis discussion either way).

---

## 8. Longer-horizon / stretch ideas (not urgent, worth noting so they aren't lost)

- **Online/streaming setting**: everything here is trained and evaluated on complete, already-finished pipeline runs. A more ambitious extension: can the model localize a fault *while the pipeline is still running*, using only the nodes executed so far (a partial graph)? Would need a different training setup (masking future nodes) but reuses the same underlying features.
- **Transfer to a genuinely different pipeline task** (not HotpotQA-style QA) — testing whether the uncertainty-propagation signal generalizes across task domains, not just across topology shapes within the same task. A much larger undertaking (new question sets from a different domain, likely new calibration), flagged here only so it's not forgotten as a "if there's time" idea.
- **Human-in-the-loop verification of "hard" cases**: for trials where `verification.target_deviated: false` (fault was injected but didn't measurably show up) or where the model is genuinely uncertain, consider whether a small human-annotated subset could serve as an additional, higher-trust test set beyond the automatically-verified labels.

---

## Suggested next-step priority (if picking just a few)

Given limited remaining time on a thesis timeline, the highest-value items to actually pursue (in order) are likely:
1. ~~Bidirectional edges ablation (Section 6)~~ — **done, 2026-09-27.** Real, cheap win (ood_top1 0.206→0.232). See `docs/model/model.md` Section 8 for the full experiment log.
2. ~~Non-graph ML baseline~~ — **implemented** (`model/nongraph_baseline.py`), not yet run against the final winning architecture. `diagnose_trace` baseline still not attempted.
3. ~~Per-topology OOD breakdown~~ — **done, 2026-09-27** (`model/eval_ood_breakdown.py`). Finding: weakness is spread fairly evenly across all 6 OOD topologies (0.15–0.32 range), not concentrated in one bad topology — points at a systemic feature/signal limitation, not a topology-coverage gap. Significance testing (McNemar's) still not done.
4. Multi-fault dataset extension (Section 1) — still the single most substantive *new* contribution on this list if time allows, since it goes beyond what the current single-fault dataset can even test. Not started.

**Since this list was written**, the following were also tried (see `docs/model/model.md` Section 8 for full numbers): node embeddings (raw 384-dim — overfit badly at first), input-projection + dropout regularization + PCA-32 compression (fixed the overfit, real OOD gain), a hyperparameter sweep (hidden_dim=64/num_layers=2/lr=1e-3 won), and a multi-task fault-type auxiliary head (neutral/slightly negative, not worth keeping as implemented). None of these were anticipated in the original list below — richer *features* (Section 2) turned out to matter more than architecture tweaks (Section 6) alone, though both mattered together (a regularized architecture was necessary to actually benefit from richer features, not sufficient on its own).

Best OOD top1 reached so far: **0.275** (still well short of the 0.6 target this round aimed for). The two levers expected to close that gap — k=10 self-consistency regeneration (up from k=5) and re-running the full enrichment pipeline on that richer data — were in progress as of this writing, not yet complete.

Everything else here is valuable but lower-urgency — worth documenting now (this file) so it isn't lost, without treating it as blocking the first model's results.
