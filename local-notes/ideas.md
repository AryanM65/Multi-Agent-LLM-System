# Next-round ideas: brainstorm from related papers and from our own results

Written 2026-11-12 (evening). LOCAL ONLY (`local-notes/`, excluded via `.git/info/exclude`). Never commit or push. Companion files: `file.md` (story), `context.md` (technical reference), `paper_review.md` (AEGIS paper review).
Plain-language summary of the state when this was written, then the idea list with sources, ranking and the agreed plan.

## 0. State of the project when this brainstorm started (so it can be resumed after a context reset)
- Task: failure attribution on AEGIS-Bench (predict faulty `(agent, mode)` pairs; 14 modes FM-1.1..FM-3.3), plus out-of-domain Who&When (184 records, exactly 1 faulty agent each).
- All planned loop steps are DONE and the loop is deleted; no Kaggle kernel is running. Nothing is committed or pushed (user rule: not until told). Many uncommitted files exist (see `file.md` last entry).
- Best results (grouped split, clean encoder early-stopped on a held-out slice of train, 2 encoder seeds, 5 downstream seeds; micro pair / agent / error):
  - AEGIS test: floor 7.35 / 68.58 / 27.14; two-stage flat MLP 26.43 / 81.19 / 38.57; 2-layer GNN 27.30 / 81.89 / 39.16; **name-embedding flat MLP 29.41 / 79.78 / 39.09 (best)**.
  - Who&When (zero-shot, floor 3.42 / 38.32 / 13.33): base 4.16 / 43.00 / 11.15; name-free 5.29 / 45.48 / 12.74; **name-embedding 5.04 / 40.85 / 15.04 (best overall)**. Published: DCL 1.60 / 8.40 / 14.67; Aegis-SFT 4.03 / 51.14 / 9.87; o3 7.41 / 53.10 / 14.88.
  - End-to-end graph fine-tune (1 seed): name-dependent graph 23.69 vs control 26.00 (graph worse); name-free graph 22.90 vs control 20.19 (graph better): sign flips, so no stable graph effect.
- What produced the improvements: (1) fine-tuning the text encoder (largest: +7 to +9 pair with the earlier encoder, +6.6 with the clean one), (2) agent-name text embedding (+3.0 pair in-domain, replicated on both encoders; Who&When error 15.04), (3) dropping name features (Who&When only, costs 1-4 points in-domain). The graph gave at most about +1 pair (not proven). Frozen-encoder graph gain came entirely from the query node.
- Key analysis findings that motivate the ideas: injected faults leave tell-tale phrases (part of the score may be injector-style recognition); mode labels often do not match the visible symptom (so mode identification has an unknown ceiling); symptoms propagate to other agents (fault propagation); some faults are invisible in text; Who&When mode mix is skewed (FM-2.6 26%, FM-3.3 21%, FM-2.2 12.5%, FM-1.1 11%) while AEGIS is balanced; 20 of 184 Who&When labels name an agent absent from the turns (max agent recall about 89%).
- Facts measured for the sibling idea: in the TEST set, 273 of 600 records share a question with another test record and 173 share both question and framework; in TRAIN, 2,841 (query, framework) groups with mean size 2.52 and 82.8% of records in groups of size 2 or more.

## 1. Related work found by web search (snippets only, not full papers; claims are the authors' own)
| Work | Core idea | Link | Status for us |
|---|---|---|---|
| AgenTracer (ICLR 2026) | Counterfactual replay and programmatic fault injection to label failures (TracerTraj, over 2,000 trajectories); 8B model (Qwen3-8B) trained with RL with reward = format + agent-level correctness + step-level Gaussian-kernel proximity | https://arxiv.org/abs/2509.03312 | published; related work, ideas for reward shaping and synthetic data |
| GraphTracer | Information Dependency Graph (DAG of information pieces); root cause = node whose correction fixes the failure; RL-trained 8B tracer; ablation says removing the IDG hurt most | https://arxiv.org/pdf/2510.10581 | preprint; related work only |
| FALAT | Attribution as dependency-guided search, since later steps can look wrong only because an earlier state was corrupted | https://huggingface.co/papers/2606.00765.md | preprint |
| AFANet ("Beyond LLM-Based Reasoning: Lightweight GNNs for Agent Failure Attribution") | Lightweight GNN on step-level semantic signals and agent-level relationships; claims to match or beat LLM baselines with far fewer parameters | https://arxiv.org/pdf/2608.18575 | preprint; the closest to our model; cite as related work |
| DCFA, CHIEF, ECHO, StepFinder, Adaptive Influence Graphs, ErrorProbe | Causal/dependency graphs, hierarchical causal attribution, consensus voting, per-step error scoring, influence graphs, verified-memory diagnosis | (see search results: arXiv 2609.04749 DCFA, 2602.23701 CHIEF, 2608.24361 AIGs) | preprints / related work |
| Who&When (ICML 2025) | Benchmark; all-at-once and step-by-step LLM judging; best agent accuracy 53.5%, step accuracy 14.2% | https://arxiv.org/abs/2505.00212 | published; our out-of-domain test |
| AEGIS / DCL (ICLR 2026) | Our dataset; DCL losses: classification (asymmetric BCE) + supervised contrastive (0.4) + hierarchy consistency (0.6), MIL attention over top-3 turns, agent/error prototype banks | https://arxiv.org/abs/2509.14295 | published; our dataset and the baseline to compare with |
Plan rule (new-plan.md): compare only against published work; the preprints (AFANet etc.) are related work, never a baseline to beat. The sourcing here comes from search snippets, so verify before citing.

## 2. The idea list (S = from a paper, M = my own suggestion or standard practice)
Tier 1 (cheap, mostly CPU; do first)
1. [S: DCL] Pair-level training with DCL-style losses: hierarchy consistency (pair prob <= min(agent, mode)), asymmetric loss, a pair head p(agent) x p(mode | agent) trained on pair targets. Now we train 14 mode logits + 1 agent logit with plain BCE and multiply at decode.
2. [M] Peer-contrast features: agent vector minus the trajectory mean, and minus the query (stronger version of the relational probe, which gave +0.6).
3. [M, novelty NOT verified] Sibling contrast: compare an agent's text with the same-role text of sibling records with the same question and framework (faulty agent should deviate from the consensus). Transductive (uses other test inputs) and benchmark-specific; must be labelled as such. Covers about 29% of test records (173 of 600).
4. [S idea from GraphTracer/FALAT, method mine] Real content-flow edges: link a turn to earlier turns whose content it copies or quotes (text overlap), replacing the guessed sequence edges; add an "originator" feature (first turn introducing new content). Our edge ablation found guessed seq/same-agent edges useless, so this is the graph's best remaining test.
5. [M] Handcrafted mode detectors: repetition (FM-1.3), low overlap with history (FM-1.4), stop words (3.1), verification keywords (3.2/3.3).
6. [S: DCL/MIL] Attention pooling over turns instead of mean pooling.
7. [M, standard] Per-class and per-framework thresholds (now only two global thresholds tuned on val).
8. [S idea: CHIEF/ECHO] Coarse-to-fine: auxiliary head for the mode group FM-1/2/3.
Tier 2 (moderate GPU)
9. [M, standard] Ensembling over encoder seeds and model types (seed noise is 1-3 points, so averaging is the most dependable gain; also flat + GNN ensemble).
10. [M] Longer-context encoder or chunk pooling (we cut each message at 256 tokens).
11. [M] Agent-name randomisation during training (shuffle names so the model cannot over-rely on them; aimed at Who&When while keeping in-domain accuracy).
Tier 3 (large)
12. [S: Aegis-SFT/AgenTracer] LoRA-fine-tune a small LLM (0.5B-1.5B) as a classifier over the whole transcript (Aegis-SFT 14B only gets 16.6 pair micro, and our small encoder already beats that).
13. [M] Per-turn perplexity under a small LM (may only recognise injector style; no help on Who&When).
14. [S: AgenTracer] RL with a step-level reward: too expensive for our hardware.
Who&When specific
15. [M, classic ML: Saerens et al. 2002] Label-shift correction: estimate Who&When's mode mix from our predictions with EM and re-weight, using no labels (targets the FM-2.6/FM-3.3 skew).
16. [M] Few-shot calibration / cross-validation on the 184 records: a separate row, no longer zero-shot.

## 3. Assessment and agreed order
- Step 0 (agreed first): a 10-minute CPU diagnostic on the current best model: pair F1 if the faulty agents were perfect, and if the modes were perfect. Expectation (unverified): modes are the bottleneck and may have a ceiling because mode labels often do not match the visible symptom.
- Recommended order: idea 1 (pair-level head + consistency loss), then 2, 6, 7 together, then 3 (sibling contrast as features first), then 4 (content-flow edges, the main graph experiment), then 15 (label shift for Who&When), then 9 (ensembling) on the best configuration; 12 only if the cheap steps saturate.
- Graph policy: implement each idea in the shared pipeline so the flat MLP and the GNN both get it; always keep the 0-layer control; only a gap over the control counts as a graph effect. Real content-flow edges (4) are the natural graph test; sibling contrast (3) as plain features first, graph variant second; peer contrast (2) as a subtraction feature (an agent-level graph could do it by aggregation); GAT-style readout for idea 6; model-independent items (1, 7, 15, 9) apply to both.
- Not prioritised: 12, 13, 14.
- Honest expectations: each step is probably +1 to +3 pair points, about the size of seed noise, so several seeds are needed; the thesis value is clearly argued results with floors and caveats, not one big number.
- Risks: sibling contrast is transductive; idea 5/13 may only exploit injector style; the graph arm doubles the work per idea; claim "novel" only after a literature check (offered for ideas 3, 4, 15).

## 4. Still open (outside this list)
Guide approval of the dataset switch; commit/push (not until the user says); write-up with the final tables beside the published rows (`aegis/docs/final_results.md` already has the clean-encoder, Who&When and end-to-end graph sections); paper-side puzzles (agent scores below floor; our zero-shot agent micro 49.2 vs 27.6); a tell-tale-phrase robustness test (not run); more encoder seeds.

## 5. Second brainstorm (2026-11-12 night): papers matched to our situation and the novelty candidates
Sources were web-search snippets plus arXiv abstract pages only (methods sections NOT read); claims are the authors' own; verify before citing.
Papers and what to take from them
- C-HMCNN / max-constraint loss (NeurIPS 2020, hard hierarchy consistency by construction); HCAL (arXiv 2508.13452, prototype contrastive + adaptive loss weighting, violation-rate metric): pair head with pair prob = min(agent, mode) built in, plus a violation-rate metric.
- Attention MIL (Ilse et al. 2018), TransMIL, Set Transformer, TransformerABMIL: replace mean pooling of an agent's turns with gated attention or a small transformer over turns.
- BBSE (Lipton et al. ICML 2018, arXiv 1802.03916), EM prior adjustment (Saerens 2002), 2025 revisit of EM: label-shift correction for the Who&When mode mix. AFANet's abstract already mentions inexpensive test-time adaptation on the OOD benchmark, so this is NOT novel for us.
- Shortcut / spurious-correlation work (Kirichenko last-layer retraining, Liu et al. error-based groups, UNMASK 2608.09209, group DRO, IRM): relevant because injected faults leave tell-tale phrasing (injector-style shortcut).
- Counterfactual / paired contrastive (SupConFL contrasts faulty code with its fix; PairCFR): the clean-twin idea that DCL uses and we lack.
- Same-field preprints: ASCon (2608.10646: direction-aware graph attention, step-to-agent attention, agent-conditioned step contextualization; claims faulty-agent, faulty-step and failure-mode gains), DuoTrace (2608.29646: detect-before-attribute with a VAE anomaly detector trained on normal runs), EMFA (2610.11600: error-propagation modelling + counterfactual verification, Who&When only), AFANet (2608.18575), DCFA, AIGs, StepFinder. Graph-based attribution is a CROWDED space, so our GNN is not a novelty claim.
Novelty candidates built from our situation (none implemented; novelty NOT verified)
- A. Sibling-derived pseudo-clean reference: records with the same question and framework come from the same underlying task, and in each sibling a different agent set is faulty. An agent's text in the siblings where it is not faulty approximates the clean run DCL uses as positive. Features / contrastive target = deviation of each agent's turn embeddings from the sibling consensus (leave-one-out). Covers 173 of 600 test records (29%), 83% of train. Transductive and label-free at test time; must be labelled so. Closest literature: SupConFL / paired counterfactual contrast, DuoTrace's normal-run anomaly idea.
- B. Deviation-then-de-propagation: score each agent's deviation (from A or from peers), then discount the part explained by earlier turns that copy/quote the deviating content (content-flow edges), so propagated victims (labelled non-faulty) score lower than the originator. Combines A with idea 4 and the propagation theme of EMFA/FALAT; method is ours.
- C. Leave-one-framework-out validation as a proxy for the Who&When shift, used to choose shortcut-robust training (last-layer retraining, name randomisation, name-embedding) without touching Who&When labels. Gives a principled selection rule instead of the post-hoc Who&When variant choice we made.
- Not novel, just do: attention-MIL pooling, hard-constraint pair head, per-class thresholds, label-shift correction (BBSE/EM), ensembling.
Suggested order: attention-MIL + hard-constraint pair head first (cheap, known), then A, then B, then C; literature check on A, B, C before claiming novelty.

### 2026-11-12 night: STEP 0 DIAGNOSTIC DONE (new `aegis/src/eval/diagnostic.py`, log `aegis/results/runs/diagnostic_es_s0.log`)
Setup: name-embedding flat MLP on clean encoder es_s0 vectors, 3 seeds, grouped test, thresholds tuned on val; ORACLE conditions used only to measure ceilings (not results).
| Condition (micro/macro) | pair | agent | error |
|---|---|---|---|
| model | 27.79 / 18.18 | 81.37 / 67.84 | 38.66 / 39.02 |
| oracle agents (true faulty-agent indicator, modes from model) | 31.12 / 20.53 | 92.20 / 89.25 | 40.09 / 40.57 |
| oracle modes (model mode probs masked to the trajectory's true modes, agents from model) | 56.06 / 44.54 | 72.05 / 60.95 | 79.90 / 79.67 |
- Knowing the faulty agents would add only +3.3 pair; knowing which modes occur would add +28.3 pair. MODE identification is the bottleneck, not agent detection.
- Of the faulty agents the model does find, the exact mode set is right only 21.2% of the time (323/1008, 183/1150, 183/1166 per seed).
- Caveat: the oracle-modes ceiling uses trajectory-level truth (a strong hint), so 56 is an upper bound, not a target. The `both` row is not 100 because masking multiplies the model's mode probability and thresholds still apply.
- Consequence for the plan: prioritise ideas that sharpen WHICH mode, not WHO: attention pooling over turns (find the symptomatic turn), mode-specific detectors, label graph / hard-constraint pair head, sibling deviation (shows what changed), de-propagation. Agent-level ideas (peer contrast for who) matter less for pair F1. Who&When agent level is a separate problem (name shift).


## 6. Plain-language guide to every approach (simple words; written 2026-11-12 night)
How to read this: each idea says what it does, why it could help, where it comes from (paper or ours) and its status. "Pair" means getting both the faulty agent AND its error mode right; "mode" means which of the 14 error types. Our diagnostic (end of section 5) showed that the main weakness is picking the right MODE: even when the model finds a faulty agent, it gets that agent's mode set exactly right only about 21% of the time. Knowing the faulty agents for free would add only about 3 pair points; knowing which modes occur would add about 28. So ideas that help choose the mode matter most.

### A. Ideas that are NOT new (standard, or from papers). Worth doing because they are cheap and reliable.
1. Hard pair rule (from the paper's DCL and the C-HMCNN paper). Right now we predict "is this agent faulty" and "which modes" separately and multiply the two numbers afterwards. This idea builds the rule "a pair can never be more likely than its agent or its mode" into the model and trains on pairs directly. Why it helps: the two halves stop disagreeing. Status: not done.
2. Attention pooling (from multiple-instance-learning papers and DCL). Today we average all of an agent's messages into one vector, so one bad message gets diluted by many normal ones. Attention lets the model put more weight on the few suspicious messages. Why it helps: a fault usually shows up in one or two messages, and spotting the symptomatic message tells us the mode. Status: not done. High priority because modes are the bottleneck.
3. Per-class and per-framework thresholds (standard). We currently use two cut-off numbers for everything. A separate cut-off for each mode (and maybe each framework) lets rare or hard modes be predicted when they deserve it. Status: not done. Very cheap.
4. Peer-contrast features (ours; a small extension of the relational features that gave +0.6). For each agent we add "how different is this agent's text from the other agents in the same conversation, and from the question". A faulty agent often says something that does not fit. Status: not done.
5. Mode-group helper task (idea from hierarchical papers such as CHIEF and ECHO). The 14 modes fall into 3 groups (specification, inter-agent, verification). We first predict the group, which is easier, and use it to help predict the exact mode. Status: not done.
6. Handcrafted mode detectors (ours). Simple rules turned into numbers: repetition of earlier text, very little overlap with the history, early stop words, verification words like "verified". Why it helps: some modes leave clear surface signs. Risk: may only exploit how the injector writes, so it may not help Who&When. Status: not done.
7. Ensembling (standard). Train several encoders and models and average them. Our results move 1 to 3 points between seeds, so averaging is the most reliable way to raise and steady the numbers. Status: not done; needs more encoder seeds.
8. Longer-text encoder (ours). We currently cut every message at 256 tokens, so long messages lose their ends. Use an encoder that reads more, or split a message into chunks. Status: not done.
9. Name shuffling during training (ours). Randomly swap agent names while training so the model cannot lean on specific names. Aimed at Who&When, whose agent names are different. Status: not done.
10. Correcting for Who&When's different mode mix (classic label-shift methods: Saerens 2002, BBSE 2018). Who&When has many FM-2.6 and FM-3.3 cases but AEGIS is balanced. We estimate the new mix from our own predictions, without using any Who&When labels, and re-weight. Note: AFANet's abstract already mentions test-time adaptation for the out-of-domain set, so this is useful but not new. Status: not done.
11. Label graph (ML-GCN style, standard). A small graph over the 14 modes, linking modes in the same group and modes that tend to occur together, so the model learns that related modes look alike. Helps confusable and rare modes. Status: not done.
12. Few-shot calibration on Who&When (standard). Use a few of its 184 records to tune. It would be a separate row because it is no longer zero-shot. Status: not done.
13. LoRA fine-tuning of a small language model (from Aegis-SFT and AgenTracer). Train a 0.5B to 1.5B model to read the whole conversation and answer. Big, slow and risky on our GPUs, so it is a side bet only. Status: not done.
14. Reinforcement learning with step rewards (from AgenTracer). Too expensive for us. Skipped.
15. Per-turn perplexity (generic). Check how "unnatural" each message is to a small language model. Probably only recognises the injector's style. Low priority.

### B. Ideas that could be NEW (our own proposals; none checked against the literature yet)
A. Sibling pseudo-clean reference.
   Simple version: the dataset contains several copies of the same task, each with a different agent broken. If an agent is NOT broken in a sibling copy, its messages there show what the agent "normally" says. Comparing each agent's messages with those clean-ish siblings shows what changed, which points to the mode as well as the agent.
   Why it could help: the paper's own DCL model uses a clean twin of each conversation, which the data does not give us. Siblings can stand in for it. This aims straight at the mode problem.
   Limits: it only works when siblings exist (about 29% of test records, about 83% of train); it uses other test inputs (never labels), so it must be described as transductive; we do not yet know whether anyone has done it.
   Status: not started. Best novelty bet.
B. Deviation then de-propagation.
   Simple version: a fault spreads. After one agent says something wrong, later agents copy or build on it and look wrong too, but they are not the injected ones. Step 1: measure how far each agent is from its reference (idea A, or the peers). Step 2: find which earlier message each later message copies from (content-flow edges, found by text overlap) and subtract the part that is just inherited. The agent whose wrongness is NOT inherited is the originator.
   Why it could help: it separates the cause from the victims, which fault propagation makes hard.
   Relation to papers: the propagation theme appears in EMFA and FALAT; the way we do it (text-overlap edges plus subtracting inherited deviation) is ours.
   Status: not started; depends on A and on content-flow edges.
C. Leave-one-framework-out validation.
   Simple version: AEGIS has six frameworks. Train on five, test on the sixth, and repeat for all six. A model that holds up on an unseen framework is more likely to hold up on Who&When. We use this to CHOOSE the recipe (names or no names, shuffling, robustness tricks) instead of choosing after looking at Who&When, which is what we did before and which is a mild form of selection on the test set.
   Why it matters: it gives an honest way to select methods for the out-of-domain benchmark, and the selection procedure is itself a methodological contribution.
   Status: not started; needs about six small fine-tunes, which run well in parallel on several GPUs.
D. Cross-record sibling graph (the GNN form of A).
   Simple version: draw a graph whose points are agents from the sibling records and whose lines connect the same role across siblings. Message passing then computes "how different is this agent from its siblings" automatically, and can also compare final answers (a faulty agent often ends with an answer that disagrees with the siblings' majority).
   Why it could help: all earlier graph tests drew lines inside one conversation using guesses, and gained nothing. This graph carries information a single conversation cannot contain. It is the best remaining test of whether a graph helps at all.
   Status: not started. Always run with the 0-layer control next to it.
E. Real content-flow edges (idea from GraphTracer and FALAT; our way of building them).
   Simple version: replace our guessed lines (just "next message" and "same agent") with lines that connect a message to the earlier messages it actually quotes or repeats, with direction. Used inside B and D.
   Status: not started.

### C. What each idea is mainly for
- Helps choose the mode (most valuable): 2 attention pooling, 5 group helper, 6 detectors, 11 label graph, A siblings, B de-propagation, D sibling graph.
- Helps find the faulty agent: 4 peer contrast, B, E.
- Helps Who&When specifically: 9 name shuffling, 10 label shift, 12 few-shot, C leave-one-framework-out; the name-embedding variant we already have is the best Who&When setting so far.
- Makes everything steadier: 7 ensembling, 3 thresholds.

### D. Suggested order (matches sections 3 and 5)
Cheap and reliable first: 2, 1, 3, 4 on the cached vectors. Then the novelty bets: A, then D and E, then B. Then C and 10 for Who&When. Ensembling (7) last, on the best setup. 13 only if time remains. Run every graph idea together with its 0-layer control, and use at least 5 seeds because differences of 1 to 3 points are within noise.

### E. Honest status of novelty
Nothing in this file has been confirmed as new. A, B, C, D and E are our own proposals built from this dataset's structure; a literature check is still needed before claiming novelty. Graph-based attribution in general is already crowded (AFANet, ASCon, EMFA and others), so a graph alone is not a contribution; a graph that carries new information (D) is the only graph variant worth testing.

## 7. Brainstorm round 3 (2026-11-12 23:10, Who&When focus; sources are search-result snippets only, NOT read in full, so treat as leads)
- Escalation ladder if the cheap Who&When ideas stall (agent still below about 47 and pair below about 7.5 after ensemble, attention pooling and name-shuffling): (1) sibling counterfactual-pair training, (2) deviation then de-propagation, (3) small language-model reader on Who&When, (4) content-flow graph edges.
- Lead A: counterfactual pairs as an invariance constraint. Snippet: "Noisy Counterfactual Matching" (arxiv 2505.24843) adds a constraint built from differences between counterfactual pairs so the model ignores spurious features. Fit to us: AEGIS makes each faulty record from a successful run, so clean/faulty siblings are natural counterfactual pairs. Idea: train so the agent score depends on what changed between sibling and faulty record, not on style or names. Strongest novelty candidate; needs a real read of the paper and a literature check.
- Lead B: targeted, not generic, augmentation. Snippet (arxiv 2302.11861): randomise only the spurious domain-dependent features and keep the robust ones; generic augmentation does poorly out of domain. Fit: our name-dropout is one such feature; extend to the other suspicious features: framework one-hot (already zeroed), turn-count scalars, position rank, share of turns. Cheap CPU test: scalar-dropout copy in the doubled training set.
- Lead C: counterfactual generation for shortcuts (AutoCAD, arxiv 2211.16202). Fit: generate name-swapped and role-swapped copies of training records (swap agent names inside the text, not only in the name feature). Needs text edits then re-embedding (GPU encoder run); this is the "name-shuffle encoder".
- Lead D: test-time adaptation on the 184 unlabeled Who&When records (multi-label TTA with bound entropy minimisation, ICLR 2025, arxiv 2502.03777; plus label-shift priors from section 2). It uses unlabeled target text, so it is transductive: report it as a separate row, never as the zero-shot row.
- Lead E: dependency-guided search (FALAT, 2026 preprint, alphaxiv 2606.00765) supports the de-propagation idea: follow dependencies between turns to find the responsible agent.
- Lead F: new data. Snippets mention "Who&When Pro" (12,326 labelled failure traces, 2026) and TraceElephant (ACL 2026, arxiv 2604.22708). If public, Who&When Pro could serve as a larger Who&When-style evaluation (more reliable than 184 records) or as extra training data (separate row, since it breaks the zero-shot protocol). Availability and licence not checked.
- Lead G: leave-one-framework-out validation as the domain-generalisation recipe selector (standard; Domain Generalization, causal perspective arxiv 2209.15177).
- Sources (links from the search results): arxiv.org/pdf/2505.00212 (Who&When), alphaxiv.org/abs/2606.00765 (FALAT), arxiv.org/html/2604.22708 (TraceElephant), arxiv.org/pdf/2211.16202 (AutoCAD), arxiv.org/html/2302.11861v3, arxiv.org/html/2505.24843v2, arxiv.org/pdf/2502.03777, arxiv.org/pdf/2209.15177.

## 8. Data checks before the complex rung (2026-11-12 23:35, measured on the grouped train split)
- Sibling records are NOT edits of one clean run. Records with the same query and framework (4,172 sibling pairs checked) share on average only 7.5% of their turn texts (median 0%), only 2% of pairs share more than half, and identical prefixes cover about 6% of turns. There is no clean record in the data (0 records without labels). So the "sibling pseudo-clean reference" and "sibling counterfactual pair" ideas cannot be done as literal paired differences; they would only give task-level context, and Who&W has no sibling structure to transfer it to. Rung 1 of the escalation ladder is therefore dropped in its literal form.
- AEGIS records have 2 to 4 faulty agents each (1 faulty agent in only 324 of 3,865; 2 in 1,676; 3 in 1,495; 4 in 370), Who&W has exactly 1. This is a second shift besides the mode mix: the model is trained to expect several faults per record.
- Extra raw fields exist but are label-side: injection_strategy has only 3 values (response_corruption 8,933, prompt_injection 7,872, prompt 202) spread evenly across modes, so it is too coarse to help as an auxiliary target; malicious_action_description (text of the injected fault) and the correct answer are also label-side.
- Who&W has 142 distinct queries in 184 records, so it has repeated tasks, but not at AEGIS scale.
- Consequence: remaining complex options are within-record ones: (a) a record-level "number of faulty agents" correction so the model expects one culprit (decoding top-1 was worse earlier, but a learned count prior or calibration per record has not been tried), (b) the small language-model reader, (c) content-flow / de-propagation within a record. Option (a) is cheap and targets the AEGIS-versus-Who&W shift directly.

## 9. Brainstorm round 2026-11-13 11:20 (one standard WebSearch; snippets only, no numbers)
- FALAT (alphaxiv 2606.00765, May/June 2026): attribution as dependency-guided search over the trajectory; related to our de-propagation / content-flow-edge ideas (originator vs repeaters). Cost: needs an LLM; we only have encoder heads. Could borrow the idea as a CPU feature: mark which agent first introduced content later copied by others.
- CHIEF (alphaxiv 2602.23701): hierarchical causal-graph failure attribution; same family as de-propagation; we judged hierarchical mode heads dead for Who&W.
- TraceElephant (ACL 2026): full traces help find the precise step; Who&W Pro (12,326 labelled traces with agent, step, mode): a possible future in-domain training source for Who&W-like data (not used: the user's rule is no Who&W label for training).
- Takeaway: nothing new and cheap to test; the unexplored CPU idea closest to the literature is the content-flow originator feature.

## 10. Brainstorm round 2026-11-13 14:20 (WebSearch standard)
- Re-found FALAT (dependency-guided search, Jun 2026), CHIEF (causal hierarchical, Feb 2026), StepFinder (KDD'26, step-level, efficient), two-stage scope-then-localise (AAAI'26), AgenTracer (RL tracer), BugHunter benchmark (full observability +11-14 agent pts). All are LLM-reader or step-level methods; none gives a cheap test on our encoder-vector setup. Nothing new beyond section 9.
- Per-record fault-count correction = the --top1 decode already tested (wbtop1): +0.7 pair no-shift, nothing with shift, agent -4 to -8. Closed (dead).

## 11. Brainstorm round 2026-11-13 ~17:20 (WebSearch standard)
- New names: CatchBench (arXiv 2608.22808, "when can an agent failure be caught"), Who&When Pro (ICML/NeurIPS 2026 listings; 12,326 failure traces with agent/step/mode labels, LLMs do not saturate it), AgenTracer-8B (RL-trained small attributor, claims +18% over Gemini-2.5-Pro on Who&When). No embedding-classifier transfer paper found. Nothing cheap and new to test on our vectors; Who&When Pro would be a bigger out-of-domain test set if its data were public (not checked).
