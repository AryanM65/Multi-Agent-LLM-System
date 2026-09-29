# Model Improvement Spec — reaching `ood_top1` ≥ 0.6

> Written 2026-09-29. Current best on the full 1785-record k=10 set: **`ood_top1` = 0.411** (disc-train, hybrid pooling + 5-seed SAGE ensemble). Target is 0.6. This file lists every practice that needs to be applied to get there, in priority order, with the evidence behind each one. Numbers below were recomputed directly from `dataset/trials_k10_disc.jsonl` on 2026-09-29 unless stated otherwise.

---

## 0. Summary

The 0.41 ceiling is mainly a **data/signal problem, not a model-capacity problem**. Three facts show it:

1. **Noise faults are close to invisible, even with a perfect counterfactual.** Every faulted trial has a clean control with the same question and topology (1532/1532). Comparing each node against that paired clean run and picking the most-changed node localizes noise only **0.19–0.21** of the time — about random (0.199). With a noise temperature of 1.2 the fault barely changes the target. Noise is 204/632 (32%) of the OOD set.
2. **Many faults have no observable effect at all.** The same paired-clean "first deviating node" oracle only reaches **0.545** on contamination and **0.624** on ceiling. The other ~40% of those trials look like clean runs at the target node. `verification.target_deviated` agrees: 27% noise, 29% contamination, 8% ceiling.
3. **Architecture tweaks are saturated.** Regularization, GAT, PCA-64, focal loss, 7 seeds, 2 layers, hidden 128 and multi-task all landed at 0.40–0.44. Differences under ±0.04 are inside the OOD 95% confidence interval (n=632, SE ≈ 0.019).

Arithmetic of the target: 0.6 × 632 = 379 correct graphs. If noise stays at ~0.2 (≈41 correct), the other 428 graphs would need **0.79** accuracy — above the counterfactual oracle for contamination and ceiling. **So 0.6 is reachable only if the regenerated noise/ceiling data (launched in commit ea08c37) actually makes those faults observable.** Everything else in this spec amplifies a signal that has to exist first.

---

## 1. Current state (what exists)

| Item | Value |
| --- | --- |
| Dataset | `trials_k10_disc.jsonl`, 1785 records, k=10, 120 questions, Qwen2.5-7B-Instruct-AWQ |
| Split | 14 train topologies (3–6 nodes, hand-designed) · 6 OOD topologies (`ood_random_0..5`, 5–7 nodes, random DAGs) |
| Sizes | train 927 · val 226 · OOD 632 graphs |
| OOD labels | noise 204 · contamination 190 · ceiling 163 · clean 75 |
| Features | 12 base scalars + 14 discrepancy scalars + PCA-128 of mean and std MiniLM embeddings (`in_dim=794` pre-PCA) |
| Model | SAGE, 3 layers, hidden 64, dropout 0.3, bidirectional edges merged into one edge type, hybrid pooling, 5-seed ensemble |
| Best | 0.411 top-1 / 0.536 top-2 OOD |
| Selection | early stopping on val top-1 (val = held-out *questions* on *train* topologies) |

Already ruled out (do not repeat): regularization sweep, GAT, PCA-64, focal loss + 7 seeds, 2 layers, hidden 128, dataset scale-up 1278→1785, dropping embeddings, within-graph z-score/margin encoding, multi-task fault-type head.

---

## 2. Diagnosis — where the error comes from

### 2.1 Single-feature oracle hit rates (k=10, argmax within graph, random = 0.199)

| Feature | noise | contamination | ceiling |
| --- | --- | --- | --- |
| `semantic_uncertainties` | 0.176 | **0.444** | 0.240 |
| `uncertainties` (lexical) | 0.114 | 0.305 | 0.170 |
| `node_novel_ratio` | 0.367 | 0.192 | 0.026 |
| `node_child_novel` | **0.375** | 0.095 | 0.085 |
| `node_dropped_ratio` | 0.176 | 0.319 | 0.302 |
| `node_sibling_disagree` | 0.124 | 0.378 | 0.284 |
| `node_len_std` | 0.259 | 0.274 | 0.049 |
| −`node_len_z` | 0.176 | 0.299 | **0.716** |
| Paired clean run, most-changed node | 0.189 | 0.443 | 0.451 |
| Paired clean run, first deviating node (DAG-aware) | 0.205 | 0.545 | 0.624 |

Readings:

- Every fault type has a *different* best feature. The model has to learn "which fault type is this, then which feature matters" — a mixture, not one score.
- The DAG-aware "first deviating node" rule beats "most changed node" by +0.10 to +0.17. **The fault is the most upstream anomaly, not the biggest anomaly.** Symptoms grow as they move downstream.
- Noise's only above-random signals (`node_child_novel`, `node_novel_ratio` ≈ 0.37) come from the node's *token profile thinning*: at high temperature fewer tokens reach the 50% agreement bar, so its children look "novel" relative to it. That is a weak, indirect signal.

### 2.2 How faults physically propagate (from `btp-pipeline/src/pipeline.py`)

- Each node draws k samples, but downstream nodes receive **only the modal sample** (`result.output`). So dispersion never propagates; only *content* does.
- **Noise** at node N raises only N's own k-sample dispersion. Its children run at normal temperature on one (possibly odd) sample, so they look confident.
- **Contamination** at node N corrupts N's *input* (the parent's text as seen by N). N's parent is clean; N's output has no origin in its parent's real output.
- **Ceiling** at N truncates N (shorter, more self-consistent output). Uncertainty goes *down*.

Implication: the useful signals are **per-edge** (does child content follow from parent content?) and **per-node against a clean baseline**, not raw per-node uncertainty.

---

## 3. Practices — data generation (`btp-pipeline/`)

### 3.1 P0 — Make noise and ceiling observable (in progress, verify before training)

The regeneration launched in `ea08c37` (noise temperature 1.2→1.8, ceiling strip 0.5→0.75, reasoner cap 2→1 step, writer cap 5→2 words) is the single most important item. Before training anything on it, run these **acceptance checks**:

- [ ] Paired-clean first-deviating-node oracle on the new noise trials ≥ 0.5 (was 0.205).
- [ ] Same oracle on new ceiling trials ≥ 0.75 (was 0.624).
- [ ] −`node_len_z` oracle on ceiling stays ≥ 0.7 and noise's dispersion features (`semantic_uncertainties`, `node_len_std`) clearly beat 0.3.
- [ ] Read 5 noise samples at temperature 1.8 by hand. They must be *degraded but on-topic*. If they are gibberish, noise becomes trivially detectable and the thesis claim weakens — drop to 1.5.
- [ ] Store the oracle numbers next to the data (e.g. `dataset/k10_partial/`) per the log-keeping rule.

If noise still fails the check, options in order: temperature 2.0 with `top_p=1.0`; inject noise as "sample from the top-k=50 tail" instead; or add a prompt-level perturbation (shuffled evidence order) as a second noise mechanism.

### 3.2 P0 — Record token log-probabilities during generation

vLLM returns logprobs for free (`logprobs=1`). Store per node: mean token logprob, mean token entropy, min logprob over the answer span. High-temperature sampling produces tokens the model itself rates unlikely, so **mean logprob is the most direct noise detector available** and needs no clean baseline. This requires the regeneration run to write the field; it cannot be recovered post hoc. Add it to the noise/ceiling regen if it has not finished, or to the next run.

### 3.3 P1 — More and more-varied *train* topologies

The 14 train topologies are hand-designed (mostly 4–6 nodes); the 6 OOD ones are random DAGs with 5–7 nodes. That is a distribution shift in size and shape. Add **10–20 random-DAG train topologies** from `dataset/generate_topology_pool.py`, sized 4–7 nodes, with a different seed than the OOD ones and a check that none is isomorphic to an OOD topology. Also add 2–3 more OOD topologies so the test estimate is steadier (6 topologies is a small sample).

### 3.4 P1 — Keep the paired clean control per (question, topology)

Every faulted trial currently has a same-question clean control. Keep that design in every future run. It is what makes the counterfactual oracle in §2.1 possible, and it enables relabeling inert trials (§4.3).

### 3.5 P2 — Fix the verifier

`btp-pipeline/src/diagnose.py` `verify_against_baseline()` uses `assumed_std=0.10`; the empirical per-node std is 0.165. Replace it with per-(topology, node) std from controls, and add a length-based and content-based deviation test (ceiling lowers uncertainty, so an uncertainty-only z-test can never mark it deviated). Recompute `target_deviated` post hoc for the existing data.

---

## 4. Practices — labels, features and data hygiene (`model/data/`)

### 4.1 P0 — Calibrate every scalar against clean controls, leave-one-out

Only `node_len_z` is calibrated today. Do the same for **every** scalar (lexical, semantic, jaccard, novel, dropped, sibling, len_std, and logprob once present): z = (value − mean over clean controls of the same `(topology_id, node_id)`) / std. Keep the raw value too.

Two correctness fixes in `compute_length_baselines()`:

- **Leave-one-out.** A clean control is currently included in its own baseline, which pulls its own z toward 0 and makes "no fault" artificially easy to call, especially on OOD topologies with ~12 controls each. Exclude the trial itself.
- **Minimum baseline size.** Fall back to a per-role baseline when a `(topology, node)` has fewer than 5 controls, with a has-flag.

This is different from the within-graph z-score that hurt OOD: that normalized nodes against each other inside one graph, destroying the absolute signal. Control-based calibration keeps it.

### 4.2 P0 — Edge-level discrepancy features

The fault signals live on parent→child edges (§2.2), but edges currently carry nothing. Compute per edge (p→c):

- Token Jaccard between p's **modal output** and c's profile (the modal output is what c actually saw; the profile is only an approximation).
- Cosine between p's modal-output embedding and c's mean embedding.
- Final-answer agreement: does c's majority final answer match p's (for reasoner/writer)?
- Length ratio len(c)/len(p), calibrated against controls.
- Optional, strongest: NLI entailment score p→c with a small NLI model (e.g. a DeBERTa-v3 MNLI checkpoint), post hoc on CPU/GPU.

Feed these as `edge_attr` (see §5.1). Also aggregate them to nodes: max/mean incoming discrepancy, max/mean outgoing discrepancy.

### 4.3 P0 — Handle inert fault labels (train side only)

~40% of contamination/ceiling trials and ~80% of current noise trials show no change at the target node against their paired clean run. Training on them teaches the model to guess.

- **Train/val:** drop trials whose target node is not measurably different from its paired control, or down-weight them (weight = normalized paired distance). Log how many are dropped.
- **OOD test: never filter.** Report the headline `ood_top1` on the full OOD set. Also report a secondary `ood_top1_effective` on the subset with observable faults, clearly labeled.

Stating it this way in the thesis is defensible: a fault with no observable effect is not localizable from the outputs by any method.

### 4.4 P1 — Topology-position features

Only a role one-hot describes position. Add scale-free structural features that transfer to unseen shapes: in-degree, out-degree, depth / max depth, number of descendants / N, is_source, is_sink, number of parents with high calibrated anomaly. These let the model learn "most upstream anomalous node" (§2.1) without memorizing topologies.

### 4.5 P1 — Replace raw embeddings with relational embedding scalars

PCA-128 of mean and std embeddings is 256 dims of mostly question content, which is what drives the train/OOD gap. Keep an ablation with them, but add compact, question-agnostic scalars:

- Mean pairwise cosine among the k samples (embedding dispersion as one number).
- Cosine of the node's mean embedding to its clean-control centroid for the same (topology, node) — a per-node "drift from normal".
- Cosine to the question embedding.

If these match the PCA-128 run, drop PCA-128: fewer dims and less question-specific overfitting.

### 4.6 P1 — Answer-level features

Extract the final answer from every sample (the `FINAL ANSWER:` line exists for reasoner). Per node: answer entropy over k, majority-answer share, and whether the majority answer changed relative to each parent's. Contamination often flips the answer; noise spreads it; ceiling collapses it to a short guess.

### 4.7 P2 — Remove dead features

`inference_gaps`, `item_frequencies`, `per_node_diagnoses` and `diagnosed_label` are `None` in 100% of k=10 records but still enter the model as 0.0 + a has-flag. Remove them (4 constant dims) to keep the input clean.

---

## 5. Practices — model and training (`model/`)

Add each item as a new numbered `model/improve_v10.py`, `v11.py`, … file, one axis per file, per the existing convention.

### 5.1 P0 — Direction-aware message passing with edge features

`build_graph_dataset.py` concatenates forward and reverse edges into one `edge_index`, so SAGE cannot tell a parent from a child. Localizing "the most upstream anomaly" needs that distinction. Replace with two relation types (forward, reverse) using `RGCNConv`, `HeteroConv` with separate `SAGEConv`s, or `GINEConv`/`TransformerConv` with an edge-type channel plus the §4.2 edge features. This is untested and is the most promising architecture change left.

### 5.2 P1 — Fault-type mixture head

The best feature differs per fault type (§2.1). Instead of an auxiliary type head (tried, neutral), make the type prediction *gate* the localization: score(n) = Σ_type P(type | graph) · s_type(n), with one small node head per type, trained end-to-end on the localization loss only (optionally with a light type loss). This uses the type information directly instead of hoping a shared head learns it.

### 5.3 P1 — Structural decoding prior

Add a DAG-aware term to the logits: score(n) += λ · (anomaly(n) − max over parents of anomaly(p)), with λ learned. It encodes "the fault is where the anomaly starts". It can also run as a standalone non-learned baseline (the classical, belief-propagation-style comparison asked for in `futurework.md` §4).

### 5.4 P1 — Model selection by leave-topology-out, not by question

Val today holds out questions on the *same* 14 topologies, so early stopping and every sweep choice optimize in-distribution accuracy. Also, the ledger picked winners by looking at OOD, which slowly overfits the test set. Replace with:

- **Grouped k-fold by topology** on the train topologies (e.g. 5 folds, ~3 topologies held out per fold). Early stopping, hyperparameters and feature choices use the mean held-out-topology accuracy.
- **OOD is scored once per final recipe**, not per sweep point. Record in `metrics_history.jsonl` which runs were selection runs.

### 5.5 P1 — Strong tabular baseline and stacking

With ~1000 training graphs, a gradient-boosted per-node ranker (LightGBM `lambdarank`, one group per graph, with the no-fault option as a virtual node) on the calibrated + edge-aggregated + structural features is likely competitive. Build it as a required baseline (the "is the graph doing work?" question for the thesis), then average its per-graph softmax with the GNN ensemble. The earlier RandomForest (0.249) used only the raw k=5 features, so it is not a fair comparison anymore.

### 5.6 P2 — Topology-level augmentation

Randomly drop a leaf node (and its features) or a non-bridging edge at train time, recomputing structural features. This exposes the model to more shapes without new generation. Only keep it if leave-topology-out accuracy improves.

### 5.7 P2 — Calibrated no-fault decision

The no-fault class is 75/632 of OOD and was the weakest role (AUC 0.675). Tune a threshold on max node probability vs the no-fault logit on the leave-topology-out folds, and temperature-scale the ensemble.

---

## 6. Practices — evaluation and reporting

- **Report uncertainty on every number.** OOD n=632, so one run's top-1 has SE ≈ 0.019 and a 95% CI of about ±0.04. Report mean ± std over ≥5 seeds, and a bootstrap CI over OOD graphs for the ensemble.
- **Test differences with McNemar's test** on paired OOD predictions before calling anything a win. Several ledger "wins" (0.395→0.403→0.411) are inside the noise.
- **Always break down** OOD top-1 by fault type, by topology, and by target role, and log the breakdown in `log_run(...)` (`eval_ood_breakdown.py` exists; call it from every entrypoint).
- **Log the oracle table** (§2.1) for every new dataset version, so feature gains can be judged against what the data allows.
- **Keep the headline metric fixed**: `ood_top1` on the full, unfiltered OOD set. Secondary metrics (`ood_top1_effective`, top-2, macro-F1) go beside it, never instead of it.
- **Save every Kaggle log** to `docs/results/kaggle_logs/<run>/` as soon as the kernel completes (project rule).

---

## 7. Roadmap — order of work

Expected gains are estimates from the oracle numbers, not measurements. Each step's gain is judged only by leave-topology-out CV plus a single final OOD score.

| # | Step | Where | Needs GPU regen? | Expected OOD effect |
| --- | --- | --- | --- | --- |
| 1 | Verify noise/ceiling regen passes §3.1 checks; merge with existing contamination + clean | pipeline, `dataset/` | running now | prerequisite; noise 0.2 → 0.4–0.5 if it passes |
| 2 | Leave-one-out calibrated z for all scalars; drop dead features | `enrich_discrepancy.py` | no | +0.02–0.05 |
| 3 | Edge discrepancy features + node aggregates | new `model/data/enrich_edges.py` | no | +0.03–0.06 |
| 4 | Structural position features | `build_graph_dataset.py` | no | +0.01–0.03 |
| 5 | Leave-topology-out model selection | `model/data/split.py`, new entrypoint | no | honest numbers; small real gain |
| 6 | Relation-typed conv with edge attributes | `gnn.py`, `improve_v10.py` | no | +0.03–0.06 |
| 7 | Filter/down-weight inert fault labels in train | `improve_v11.py` | no | +0.02–0.05 |
| 8 | Fault-type mixture head + structural prior | `gnn.py`, `improve_v12.py` | no | +0.02–0.04 |
| 9 | LightGBM ranker baseline + stacking | `model/nongraph_baseline.py` | no | +0.01–0.03 |
| 10 | Logprob features, more train topologies, more OOD topologies | pipeline | yes | +0.03–0.08, and a steadier test estimate |

Steps 2–9 run on existing data and can be done while step 1's regeneration finishes. If step 1's checks fail, fix the fault injection before investing in 6–9: no model can localize a fault that the data does not show.

### Definition of done

- `ood_top1` ≥ 0.60 on the full, unfiltered OOD set, 5-seed ensemble, with a 95% bootstrap CI reported.
- The recipe was chosen by leave-topology-out CV, and OOD was scored once.
- Per-fault-type OOD accuracy logged; noise ≥ 0.45 so the result is not carried by contamination alone.
- Run logged in `model/metrics_history.jsonl` and this file's §1 table updated.
