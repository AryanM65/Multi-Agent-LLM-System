# rerank_rag_context.md: context for the "model shortlist + LLM judge + AEGIS-retrieval" idea (Track A)

Written 2026-11-14 as a hand-off to Claude Code. Format rule for this project: bullets only, no tables.

(Saved verbatim from the text the user pasted into the chat on 2026-11-14; the original file was not in the repo.)

Read the whole file before touching code. Section 9 lists what to verify in the repo first. Section 10 lists the standing rules.

---

## 1. Goal and the two tracks

- **Project:** BTech thesis (BTP-I, NSUT, supervisor Dr. Vandana Bhatia) on failure attribution in multi-agent LLM systems. Base paper: AEGIS (Kong et al., ICLR 2026, arXiv 2509.14295).
- **Task:** given a failed multi-agent conversation, predict the set of (agent, error mode) pairs, using the 14 MAST error modes. Metrics: pair, agent and error micro/macro F1.
- **User's goal for this phase:** raise Who&When pair/agent/error F1 above the paper's numbers, without hurting AEGIS-Bench, while keeping the claim honest.
- **Track A (the one we are doing):** strict out-of-distribution. Train only on AEGIS-Bench. Who&When is never used for training, tuning, thresholds, epochs, retrieval index or prompt examples. It is scored once at the end with a configuration fixed beforehand.
- **Track B (NOT doing now):** training or retrieving from matched or Who&When-like data (the AgenTracer recipe, or few-shot retrieval from Who&When itself). It is a different, weaker claim. If ever run, report it separately, use leave-one-out, and guard against duplicate tasks.
- **Decision already made by the user:** fine-tuning and LoRA are not separate "approaches". We fine-tune anyway as part of the pipeline.

---

## 2. Current state (from approach.md, 2026-11-13)

- **Datasets:**
  - AEGIS-Bench: 9,533 injected-fault conversations. We use a query-disjoint "grouped" split, because 588 of 600 released test queries also appear in train. Test is the 600-record grouped test.
  - Who&When: 184 real failed conversations, human labels, free-form agent names, exactly one faulty agent per record. The AEGIS paper maps its free-text `mistake_reason` to the 14 modes with an LLM classifier. Who&When also has a decisive-step label, which our pipeline does not predict.
- **Pipeline:** fine-tuned MiniLM (L6/L12, 256/512 tokens) embeds turns. One feature row per agent (mean turn vector, scalar features, optional name block and peer-contrast block). Small MLP head, 15 outputs (14 modes + agent-faulty). Two-threshold decoder: output (agent, mode) if agent prob >= ta and mode prob >= tm, both tuned on AEGIS grouped validation. If nothing passes, output the single best pair.
- **Best methods so far:** block-dropout (four mask copies: all blocks, no names, no peer, neither), name-dropout, peer-contrast, label-shift EM (Saerens et al. 2002, transductive), ensembles of 7 or 14 encoders.
- **Best honest joint row (`wbes14` plus label-shift):** AEGIS test pair 32.96. Who&When 8.24 / 47.72 / 17.12 (pair/agent/error). Its two halves gave 8.79 and 7.76, so the realistic value is about 8.2.
- **Floors:** AEGIS 7.35 / 68.58 / 27.14. Who&When 3.42 / 38.32 / 13.33.
- **Published reference (micro F1, paper Table 1):** Aegis-SFT 14B: AEGIS 16.62/76.53/27.53, Who&When 4.03/51.14/9.87. o3 zero-shot: AEGIS 7.86/40.31/22.37, Who&When 7.41/53.10/14.88. DCL: AEGIS 8.33/22.93/24.73, Who&When 1.60/8.40/14.67.
- **Closed, do not revisit:** order-aware sequence encoder, framework-adversarial (gradient reversal), hierarchical mode heads, kNN (floor on Who&When: 2.69 pair), top-1 decode, dataset-level test-time adaptation, small LoRA generator readers.
- **Known protocol weaknesses to fix (Step 0):**
  - Many Who&When rows are logged, and the best honest row was chosen among them. Pre-commit one final configuration chosen only by AEGIS validation and leave-one-framework-out.
  - The names-only mask rows (9.07, 9.60) were chosen after seeing Who&When. Keep them out of the headline.
  - The dataset switch has not yet been approved by the project guide.

---

## 3. The idea in one paragraph

- Your trained ensemble proposes a shortlist of the top-k (agent, mode) candidates with probabilities. A frozen LLM then acts as a judge: for each candidate it answers "did agent A commit mode M?" The judge's Yes-minus-No log-probability is fused with the model's logit. A second judge variant adds retrieved, labelled AEGIS examples to the prompt (the RAG part). A threshold on the fused score gives the final pairs.
- Nothing about the LLM is trained. The LLM and the retrieval run at inference time. The retrieval index is built offline from AEGIS train only.
- Why an LLM at all: it brings knowledge from outside AEGIS and gives an independent signal. From our own numbers, o3 zero-shot has better Who&When agent F1 (53.10) than our ensemble (47.72), while our ensemble is better on error F1 (17.12 against 14.88).
- Why keep the trained model: it prunes the search (4 agents x 14 modes = 56 questions otherwise), carries AEGIS base rates, and gives calibrated probabilities.
- This is an empirical bet, not a guarantee. The ablations in section 7 decide whether each part earns its place.

---

## 4. Detailed design

### 4.1 Shortlist
- Input: the saved per-agent probabilities (faulty prob and 14 mode probs) from the ensemble, after label-shift for Who&When.
- Candidate score for (agent a, mode m): the same product or decoder score used now (agent prob and mode prob). Rank all pairs, keep top-k (try k = 3, 5, 8).
- Re-ranking can only reorder or filter candidates already in the top-k. It can never add a true pair that is outside it.

### 4.2 Plain judge (approach 4)
- Pointwise yes or no per candidate. Prompt contents, in this order: instruction, the 14 FM mode definitions (reuse the definitions in the AEGIS evaluation prompt, appendix E.3 of the paper), the task query, the conversation (truncate long turns), then the candidate question last.
- Put the conversation first and the candidate last so all k candidates of a record share one cached prefix (KV or prefix caching). One long prefill, k short continuations.
- Score: the judge logit J = log P("Yes") - log P("No") at the first generated token. Use constrained single-token scoring, no free generation.
- Order bias control: run twice with the Yes/No option order (or candidate order) swapped, and average. Test the chosen judge for bias, because small models are strongly biased and permutation does not always help.
- Judge model: any frozen open LLM that fits (7B to 8B instruct in 4-bit on a Kaggle T4 is the starting point). Disclose which model, because the paper's o3 baseline is much stronger.

### 4.3 Retrieval-augmented judge (approach 5, the RAG part)
- Index (offline): agent-level snippets from AEGIS grouped TRAIN only, each with its true labels. Trim each turn so a snippet is about 200 to 300 tokens. Embed with the fine-tuned MiniLM agent vectors.
- Retrieval is mode-first. For each candidate (agent a, mode m): embed a's snippet, then fetch the r nearest positives labelled m and the r nearest negatives where m is absent (r = 2 or 3). Positives and negatives should come from different frameworks where possible.
- The retrieved cases are shown WITH their own ground-truth labels (labels of other AEGIS train conversations, never of the conversation being judged). If the AEGIS release includes the injection plan text (what was changed), use it as a short rationale for each example. Verify that it exists first.
- Prompt order: mode definition, labelled examples, a line saying the examples come from a different system and show the pattern only, then the test snippet and the same yes or no question.
- Weak-match fallback: if the best retrieved similarity falls below a cutoff, drop the examples for that candidate and use the plain judge score.
- Difference from kNN (4.14): kNN votes over neighbour labels. Here the LLM reads the neighbours and can reject a poor analogy. kNN scored at the floor on Who&When, so transfer of AEGIS similarity is a real risk.

### 4.4 Fusion and decoding
- s = logit(p_model) + w1 * J_plain + w2 * J_rag, with judge scores z-scored within each dataset (or use reciprocal rank fusion as an alternative that ignores calibration).
- p_model is the candidate probability, logit(p) = ln(p / (1 - p)). Convert back with the sigmoid if you want a probability.
- Weights w1, w2 are tuned on AEGIS grouped validation (or leave-one-framework-out), never on Who&When.
- Decode: apply a single threshold on the fused score. Tune it on a different half of validation than the weights. If nothing passes, output the single best candidate (as now).
- Optional: a small logistic regression over [logit p_model, J_plain, J_rag, candidate rank], few parameters only.
- Worked example (illustrative): p_model = 0.30 gives logit -0.847. Judge Yes prob 0.85 gives 1.735. Retrieval judge 0.90 gives 2.197. With w1 = w2 = 0.5: s = -0.847 + 0.867 + 1.099 = 1.119, sigmoid about 0.75.
- The log-score addition is standard (naive Bayes, product of experts / Hinton 2002, log-linear opinion pooling). It is not novel. What is ours is applying it to model-plus-judge fusion on this task.

### 4.5 Model plus retriever only (no LLM) variant
- Retrieval as features: for each candidate, the share of retrieved neighbours labelled m, the mean similarity to positives minus negatives, and the distance to the nearest positive. Add as a feature block under block-dropout. Runs on CPU on saved vectors.
- Prototype scoring and a small trained matcher are further options.
- Expected to be weaker on Who&When, because nothing outside AEGIS enters the system. Run it first as the cheap test.

---

## 5. Gates (go or no-go)

- **Gate 0 (no LLM, CPU):** from the saved probabilities, compute recall@k for k = 1, 3, 5, 8 on AEGIS validation and on Who&When. Continue only if recall@5 is at least about 1.5x recall@1 (a heuristic bar of ours, not from a paper). Also report the oracle re-rank F1 (keep only true pairs inside top-k), but treat it as a loose ceiling. It assumes a perfect judge and will look far above the real 8 pair F1, so it is not a stop rule.
- **Gate 1:** the plain judge, fused, must beat the model alone on held-out frameworks in leave-one-framework-out, before Who&When is looked at.
- **Gate 2:** the retrieval judge must beat the random-examples control (same prompt, random AEGIS examples) and the plain judge on held-out frameworks.
- **Gate 3:** one pre-committed Who&When run with bootstrap confidence intervals.

---

## 6. Evaluation protocol

- **Pre-commit:** write the final configuration (k, r, judge model, prompt, w1, w2, thresholds, cutoff) to a file with a timestamp before the Who&When run. Choose it by AEGIS validation and leave-one-framework-out only.
- **Validation split:** split AEGIS grouped validation into two halves. One tunes fusion weights, the other tunes the threshold.
- **Leave-one-framework-out:** train and tune on five frameworks, score the sixth. For retrieval, rebuild the index without the held-out framework.
- **Leakage rules for retrieval:** index = AEGIS train only. Exclude snippets from the same query as the test item. Never index Who&When.
- **Noise:** Who&When has 184 records. Bootstrap (resample records with replacement, about 1,000 times, take the 2.5 to 97.5 percentile). Differences under about 2 pair points are noise.
- **Truncation:** the paper evaluates with a 32,768-token context to avoid truncating more than 95% of long Who&When trajectories. Check how our 256/512 token limits and judge truncation affect the long records.

---

## 7. Ablation rows to report (all with confidence intervals)

- Model alone (current best).
- Model plus retriever features, no LLM.
- LLM alone over all agent-mode pairs (run on a subset, it is expensive).
- LLM plus RAG alone over all pairs (same subset).
- LLM plus RAG over the shortlist only.
- Full fusion (model plus plain judge plus RAG judge).
- Random-examples control for the RAG prompt.
- Reading the results: if fusion beats LLM plus RAG alone, the model earns its place. If LLM plus RAG alone matches fusion, the trained model is redundant on Who&When and the honest claim is the retrieval-augmented judge. If the model alone matches fusion, the LLM adds cost only. Any of the three is a valid thesis result if all rows are reported.

---

## 8. Backup ideas (not yet scheduled)

- Uncertainty block: per-turn log-likelihood and entropy from a small LM, plus the instruct-versus-base log-ratio ("progress advantage"), aggregated per agent as a feature block under block-dropout. Framework-agnostic. Fits the thesis theme of uncertainty propagation.
- Label-semantic head: match agent features to embedded text definitions of the 14 modes instead of 14 opaque outputs.
- Error-first two-stage pipeline: predict the set of modes per conversation first, then score agents given each mode.
- Per-framework or per-conversation normalization of features (CORAL / AdaBN style) as a lighter alternative to adversarial training.
- Adversarial hard-negative training (mine and upweight missed injections, or regenerate harder ones if the injection code is released). Do the confusion matrix first.
- Counterfactual twin distillation: AEGIS failed conversations have clean twins. Use the twin as privileged information in a teacher and distill to a student that sees only the failed conversation. Needs the original successful trajectories in the release.
- Injector-fingerprint removal: regenerate some injections with different LLMs so the model cannot rely on the manipulator's style.
- Step prediction ("when"): not built. Who&When has a decisive-step label, and our pair F1 does not use it. Only add if the user asks.

---

## 9. Verify these in the repo or data BEFORE building

- Are the per-agent faulty and mode probabilities (after label-shift) saved for AEGIS val, AEGIS test and Who&When, for each encoder and seed? Gate 0 needs them.
- Does the AEGIS-Bench release include the injection plan text and the original successful trajectories? This decides the rationale in the RAG examples and whether twin distillation is possible.
- Is the AEGIS injection code released and runnable? It decides the regeneration ideas.
- Where are the 14 FM definitions in our code? Reuse the paper's evaluation-prompt wording (appendix E.3) so the judge sees the same definitions.
- Which LLM fits on Kaggle T4 (16 GB) at the conversation lengths we have, and whether prefix caching works there (for example vLLM, or reusing past key values in transformers).
- Does `features.py` already include a positional scalar? It affects the position-prior baseline.

(Answers found on 2026-11-14 are in local-notes/rr_runbook.md section 3.)

---

## 10. Standing rules

- Do not commit or push anything until the user says so.
- Do not use Who&When data or labels for training, tuning, retrieval or examples in Track A.
- Do not tune thresholds, weights or epochs on Who&When. Score it once with the pre-committed configuration.
- Work on copies. Do not overwrite existing result rows. Append to `aegis/results/metrics_history.jsonl` with new run names, and keep raw logs in `aegis/results/runs/`.
- Report all ablation rows, including the ones that do not help. State clearly which numbers are transductive (label-shift) and which choices were post hoc.
- Citations in this file were written from memory or from search summaries. Verify author names, years and venues before they go into the thesis.
- Windows machine, projects at `D:\aryan\Projects`. Heavy GPU jobs run as Kaggle kernels.

---

## 11. Literature behind each piece (verify before citing)

- **Shortlist then LLM rerank:** Infer-Retrieve-Rank (D'Oosterlinck et al., 2024, arXiv 2401.12178) and ICXML (Zhu and Zamani, 2023, arXiv 2311.09649).
- **Retrieved in-context examples beat random:** KATE (Liu et al., 2022, DeeLIO). Task-tuned embeddings help more, and a larger retrieval pool helps.
- **Judge plus evaluators in failure attribution:** AgentLocate (COLM 2026). Judge-Evaluator refinement with confidence-aware voting.
- **Memory or retrieval for attribution:** ErrorProbe (Findings of ACL 2026, verified memory). CORRECT (retrieval to localize the error step, venue not checked). A summary reports ErrorProbe memory gains of +10.9 agent and +13.4 step points, which is secondhand, so verify in the paper. The same literature warns that training-free judge transfer is sensitive to domain shift.
- **Judge position bias:** Zheng et al., 2023 (swap order and judge twice), and a 2026 study on rubric-based judges (balanced permutation). A small model (Qwen3-8B) picked the first option 30 to 39% of the time against an ideal 20%, and permutation sometimes lowered its correlation.
- **Cost warning:** a TU Delft bachelor thesis on scope-guided LLM judging found only a modest gain at more than four times the token cost.
- **Fusion:** product of experts (Hinton, 2002), log-linear opinion pooling (Genest and Zidek, 1986), reciprocal rank fusion (Cormack et al., 2009). All from memory.
- **Closest published attribution work:** AgenTracer (ICLR 2026, trained on injected faults), AgentRx (ICML 2026, predicts step plus category, prompt-based), XG-Guard (ACL 2026, agent-level), Huang et al. (ICML 2025, the fault-injection methodology). Preprints closest to the AEGIS task: AFANet, VerifyMAS, EDGE.
- **Base paper facts checked in its text:** Who&When is used only as an OOD test set. Hyperparameters were tuned on the Aegis validation set. The one exception is that the SFT epoch count (two) was chosen after looking at a Who&When curve. Zero-shot baselines use the same prompt on both benchmarks.

---

## 12. Honest expectations

- The shortlist-then-judge pattern is published in other tasks. The specific combination here (trained shortlist, mode-first contrastive retrieval from AEGIS, logit fusion, tested on Who&When) is our own design and was not found published as one method. A full literature check for that exact combination was not done.
- Zero-shot LLMs are weak on Who&When, and AEGIS examples may not resemble real errors. Gains may be a few points at best, and may sit inside the noise of 184 records. A clean negative result is still a valid thesis finding.
- This is the most compute-heavy idea on the list. Run Gate 0 and the retriever-only variant first.
