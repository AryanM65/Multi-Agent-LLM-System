# approach.md: every approach we tried during the improvement loop, explained in detail

Written 2026-11-13. This file explains, in simple words, what each approach is, why we tried it, how it was implemented, what it scored, and whether it is our own idea or taken from a research paper. Format rule for this project: bullets only, no tables.

Numbers come from `local-notes/results.md` (full per-run values are in its appendix) and `aegis/docs/final_results.md`. The citations were written from memory of the papers. Please verify author names, years and titles before copying them into the thesis. Only the Aegis, Who&When and MAST entries were checked against the project's own `new-plan.md`.

---

## 1. Project context (what problem we are solving)

- **Project:** BTech thesis (BTP-I, Group 121, NSUT) on failure attribution in multi-agent LLM systems.
- **Multi-agent LLM system:** several LLM "agents" (for example a planner, a coder, a reviewer) talk to each other to solve a task. The whole exchange is a "conversation" or "trajectory".
- **Failure attribution:** when the system fails, say which agent was at fault and what kind of mistake it made.
- **Prediction target:** for each conversation, a set of (agent, error mode) pairs. There are 14 error modes, such as FM-1.2 "disobey role specification", FM-2.4 "information withholding" and FM-3.2 "no or incorrect verification". They come from the MAST failure taxonomy.
- **Two datasets:**
  - **AEGIS-Bench** (Kong et al., ICLR 2026, arXiv 2509.14295): 9,533 conversations made by taking successful gpt-4o-mini multi-agent runs and injecting faults on purpose. Because the injected faults are known, the labels are exact. We train on its train split and test on its 600-record test split.
  - **Who&When** (Zhang et al., ICML 2025): 184 real failed conversations, annotated by humans. It is a different domain with free-form agent names, and it has exactly one faulty agent per record. We never train on it. We only score on it, so it measures out-of-domain generalisation.
- **Metrics:**
  - **Pair F1:** was the (agent, error mode) pair right? This is the hardest level.
  - **Agent F1:** was the faulty agent right (ignoring the mode)?
  - **Error F1:** was the error mode right (ignoring which agent)?
  - **Micro F1:** pool all predictions, then compute F1. It favours frequent classes.
  - **Macro F1:** compute F1 per class, then average. It treats rare classes equally.
  - All scores are in percent. "pair / agent / error" always means micro F1 unless stated.
- **Floor:** the score of a trivial predictor that always guesses the most common answer. AEGIS floor is 7.35 / 68.58 / 27.14, and Who&When floor is 3.42 / 38.32 / 13.33. Beating the floor is the minimum bar. Agent F1 on AEGIS is hard to read because over half the agents are faulty, so even a trivial guess scores 68.58.
- **Published numbers we compare with** (paper Table 1, micro F1):
  - Aegis-SFT, a 14B model fully fine-tuned: AEGIS-Bench 16.62 / 76.53 / 27.53, Who&When 4.03 / 51.14 / 9.87.
  - o3 (zero-shot): AEGIS-Bench 7.86 / 40.31 / 22.37, Who&When 7.41 / 53.10 / 14.88.
  - DCL, the small-model baseline: AEGIS-Bench 8.33 / 22.93 / 24.73, Who&When 1.60 / 8.40 / 14.67.
- **The user's goal for the loop:** improve Who&When pair, agent and error F1 as much as possible without hurting AEGIS-Bench test pair F1.

### Protocol we followed (important for honesty)
- **Grouped split:** in the released AEGIS split, 588 of the 600 test queries also appear in train, so many results are inflated by question leakage. We built a query-disjoint "grouped" split (no question in two sets) and train on that. This is a stricter test than the paper's.
- **Thresholds and early stopping use only AEGIS grouped validation.** The test set and Who&When are scored once. Who&When labels are never used for training or tuning.
- **Single decoder everywhere ("prod" decoder):** the model gives, for each agent, a probability that the agent is faulty and a probability for each error mode. We output the pair (agent, mode) if the agent probability is at least a global threshold `ta` and the mode probability is at least a global threshold `tm`. Both thresholds are tuned on validation. If nothing passes, the single best pair is output so no record is empty.
- **Noise:** Who&When has only 184 records, and the pair F1 changes about ±2 points between seeds. Only multi-seed ensembles are credible.

---

## 2. Glossary (complex terms in simple words)

- **Encoder / embedding:** a neural network that turns a piece of text into a list of numbers (a "vector", 384 numbers for MiniLM) so that similar texts get similar vectors. We use MiniLM, a small, fast language model.
- **Frozen vs fine-tuned encoder:** frozen means the encoder is used as-is. Fine-tuned means we also train the encoder itself on our task (here, on AEGIS grouped train), so its vectors become better at showing who is at fault.
- **MLP (multi-layer perceptron):** a small feed-forward neural network, here two hidden layers of 256 units. It takes an agent's features and outputs 15 probabilities: 14 error modes plus "this agent is faulty".
- **BCE loss:** binary cross-entropy, the standard loss when each output is an independent yes/no.
- **Early stopping:** stop training when validation loss stops improving, to avoid overfitting.
- **Seed:** the random starting point of training. Different seeds give slightly different models, so we average over several.
- **Ensemble:** averaging the probabilities from several models. It reduces noise, so it is the main way we get stable results on tiny Who&When.
- **Dropout:** randomly switching off parts of the input or network while training, so the model does not depend on any one part.
- **Distribution shift:** the test data looks different from the training data. AEGIS to Who&When is a shift (injected faults to real failures, different agent names, different error mix).
- **Label shift / prior shift:** the shift where the mix of answers changes (for example some error modes are much more common in Who&When) while the relation between text and label stays the same.
- **Transductive:** using the unlabeled test inputs themselves during prediction. Label-shift and self-training are transductive. No test labels are used, but the test inputs are looked at.
- **Mean pooling / attention pooling:** an agent speaks several times. Mean pooling averages its turn vectors. Attention pooling learns weights so important turns count more.
- **Contrastive learning:** a training loss that pulls vectors of "similar" items together and pushes "different" items apart.
- **Adversarial (gradient reversal):** a second classifier tries to guess a nuisance property (here, which framework the conversation came from) from the vector, and the encoder is trained to make that guess fail, so the vector stops carrying that property.
- **kNN (k nearest neighbours):** predict by finding the k most similar training examples and averaging their labels. No training.
- **LoRA:** a cheap way to fine-tune a large language model by training only small extra matrices added to its layers, instead of all its weights.
- **Zero-shot:** using a model on a dataset it was never trained on.

---

## 3. The pipeline all approaches share

1. **Fine-tune the encoder (GPU, Kaggle T4).** Script `aegis/scripts/kaggle_finetune_minilm.py`. MiniLM reads each turn (and the query) and is trained with a flat-MLP head, BCE loss, encoder learning rate 2e-5, head learning rate 1e-3, early stopping on grouped validation. After training, every turn and query is re-embedded and the vectors are saved (`ft_shards_*`).
2. **Build one feature row per agent.** `aegis/src/features.py` takes the mean of the agent's turn vectors, a few scalar features, and optional blocks (name embedding, peer-contrast). Rows are grouped per conversation.
3. **Train a small head on the saved vectors (CPU).** A two-hidden-layer MLP predicts the 15 outputs. Scripts in `aegis/src/eval/` (`improve_cpu.py`, `ww_focus.py`, `ww_block.py`, `ww_complex.py`, `ww_attn.py`).
4. **Decode.** Apply the two-threshold decoder above.
5. **Score** on AEGIS grouped test and Who&When, micro and macro. All runs are written to `aegis/results/metrics_history.jsonl`, and raw logs are kept in the repo (`aegis/results/runs/`, `aegis/results/kaggle_logs/`).

Offloading: GPU kernels ran on four Kaggle accounts in parallel (about 4 to 5 hours per fine-tune with 256-token input). CPU scoring was the bottleneck, so some CPU jobs were also run as Kaggle kernels.

---

## 4. The approaches, in the order we tried them

Each entry has: idea in simple words, how it was implemented, numbers, and source (novel or paper).

### 4.1 Fine-tuned MiniLM encoder plus flat MLP (the base recipe)
- **Idea:** instead of using a frozen text embedding, teach the encoder what "faulty behaviour" looks like by training it on AEGIS. The flat MLP then classifies each agent from its mean turn vector.
- **Implementation:** MiniLM-L6 (256 tokens), fine-tuned end-to-end (section 3). Later we also used MiniLM-L12 (12 layers) and 512-token inputs, and ran many seeds (`es_s0` to `es_s13`).
- **Numbers (`es_s0` with name-dropout):** AEGIS test 26.0 / 82.0 / 38.1, Who&When 6.04 / 47.04 / 14.54. In the earlier frozen-vector baseline, fine-tuning raised grouped validation pair from 22.9 to 36.9.
- **Source:** based on MiniLM (Wang et al., 2020) and the sentence-transformers library (Reimers and Gurevych, 2019). The flat-MLP-over-embeddings design is our simplified version of the DCL baseline in the Aegis paper. Fine-tuning the encoder end-to-end on this task is a standard step, applied to this benchmark by us.

### 4.2 Peer-contrast features
- **Idea:** an agent's behaviour only looks faulty compared with what the other agents did. Give the model the difference between this agent and its peers.
- **Implementation (`peer_block` in `improve_cpu.py`):** for each agent, features are (agent mean vector minus mean of all other agents' turns), (agent mean vector minus the query vector), and four cosine similarities. These are appended to the feature row.
- **Numbers:** +3.8 and +4.4 test pair on two encoders over the base recipe, with much smaller seed variance. `es_s0` with peer plus name-dropout (`peernd`): AEGIS 32.38 / 79.02 / 40.58, Who&When about 4.9 / 41.1 / 13.6. It costs 2 to 3 points of Who&When agent F1.
- **Source:** our own design. The idea of "compare an item with its context" is common in relational and contrastive features, but we did not take this specific feature set from a paper.

### 4.3 Name-dropout
- **Idea:** the AEGIS agent names (like `Solver1`) are special to that dataset, and Who&When uses totally different names. A model that leans on names cannot transfer. During training, we randomly hide the name information so the model must judge from behaviour.
- **Implementation:** the name is embedded with MiniLM and appended as a block. The training set is doubled with a copy in which the name block is set to zero, so the same model works with or without names.
- **Numbers:** Who&When agent F1 rose +6.0 and +4.2 on two encoders (mean 46.26 vs 41.14), pair 6.38 vs 5.60. AEGIS cost: about 1.8 on one encoder and none on another (alone, test pair drops to about 26 to 27).
- **Source:** feature or input dropout comes from Srivastava et al., 2014 ("Dropout"). Applying it to the agent-name block to remove a dataset-specific shortcut is our own application.

### 4.4 512-token encoder plus peer features
- **Idea:** longer input shows more of each turn.
- **Numbers:** AEGIS test 36.27 / 80.93 / 44.45 (best in-domain row), but Who&When stays at the floor (pair about 3.7 to 4.4 for the L12 512 no-names encoders).
- **Source:** our own configuration of 4.1 and 4.2.

### 4.5 Block-dropout (the main method)
- **Idea:** we want one model that is strong on AEGIS (which likes names and peer features) and also works on Who&When (where names are useless). Train it to work under any combination of those blocks.
- **Implementation (`ww_block.py`):** the training set is four copies: all blocks on, name block off, peer block off, both off. At test time, AEGIS is evaluated with all blocks on. Who&When can be evaluated under any mask. Thresholds for each mask are tuned on AEGIS validation with the same mask.
- **Numbers (7 encoders, 3 head seeds):**
  - All blocks on: AEGIS test 32.93 / 79.14 / 41.00, Who&When 6.10 / 44.57 / 12.65.
  - Names only at test time: AEGIS 27.91, Who&When 7.17 / 44.61 / 16.44.
  - Peer only: Who&When 3.85 / 40.84 / 8.09. Neither: 3.35 / 42.10 / 10.58 (floor level).
- **Source:** adapted from modality dropout (Neverova et al., 2016, "ModDrop") and the general dropout idea. The specific recipe of training four mask copies to serve both datasets is our own design.

### 4.6 Label-shift correction
- **Idea:** Who&When has a different mix of error modes than AEGIS. If the model assumes the AEGIS mix, it over-predicts the wrong modes. We can estimate Who&When's mix from the model's own predictions without labels, and re-weight.
- **Implementation (`estimate_prior` in `ww_focus.py`):** an EM loop starting from the AEGIS train mode prior estimates the Who&When mode prior. Mode probabilities are multiplied by (estimated prior divided by source prior), then decoded with the unchanged AEGIS-tuned thresholds.
- **Numbers:** 7 encoders: Who&When 8.79 / 48.54 / 17.86. The 14-encoder ensemble (`wbes14`): 8.24 / 47.72 / 17.12 with AEGIS 32.96. The shift only affects Who&When (it does not change the AEGIS decoding).
- **Source:** Saerens, Latinne and Decaestecker, 2002 ("Adjusting the outputs of a classifier to new a priori probabilities"). We implemented their EM prior estimate and applied it to this task. This is a transductive step (no test labels, but the unlabeled test inputs are used).

### 4.7 Ensembles of encoders
- **Idea:** average several models to reduce noise.
- **Implementation:** the same head is trained on each of 7 or 14 independently fine-tuned encoders and the probabilities are averaged before decoding. Thresholds are tuned on validation of the averaged probabilities.
- **Numbers:**
  - `wbes14` (14 plain encoders) plus shift: Who&When 8.24 / 47.72 / 17.12, AEGIS 32.96. This is the best honest row. Its two halves gave 8.79 and 7.76 pair, so the realistic value is about 8.2.
  - `wbmix` (10 encoders, names-only mask plus shift): Who&When 9.07 / 47.79 / 18.63, AEGIS 31.23. The mask was chosen after seeing Who&When, so this is a lead, not a claim.
- **Source:** standard ensembling (no single paper).

### 4.8 Attention pooling
- **Idea:** some turns of an agent matter more than others. Let the model learn which.
- **Implementation:** gated attention over the agent's turn vectors (`ww_attn.py` on CPU, `--attn` in the end-to-end encoder), concatenated to the usual features.
- **Numbers:** AEGIS +2.6 to +2.9 test pair, Who&When neutral. `attnsup5` (attention plus contrastive 0.5): AEGIS test 31.97, Who&When 5.52 / 45.53 / 13.06.
- **Source:** gated attention pooling from Ilse, Tomczak and Welling, 2018 ("Attention-based deep multiple instance learning"). Applying it over an agent's turns is our adaptation.

### 4.9 Supervised contrastive encoder training
- **Idea:** train the encoder so that agent vectors with overlapping error modes sit close together and others far apart. This should make the vectors more about the fault and less about incidental style.
- **Implementation (`--supcon W` in the fine-tune script):** a supervised contrastive loss, using error-mode overlap between agents as the notion of "similar", added to the BCE loss with weight W (0.2, 0.5, 1.0 were tried).
- **Numbers:**
  - Single encoders reached Who&When pair 6.0 to 7.0 and agent up to 50.2, with AEGIS 30 to 33. Several single-seed gains did not replicate on other seeds (`supcon_s1`, `supcon5_s1`).
  - Ensemble `wbsc5` (6 supcon-0.5 encoders): AEGIS 36.15, Who&When 6.01 / 47.39 / 10.07.
  - Ensemble `wbscb` (7 contrastive and attention encoders): AEGIS 36.85, Who&When 5.51 / 46.98 / 9.99. This is the best joint model for AEGIS, but Who&When pair is only about 5.5.
- **Source:** Khosla et al., 2020 ("Supervised contrastive learning"). The multi-label version using error-mode overlap is our adaptation.

### 4.10 Order-aware sequence encoder
- **Idea:** the order of an agent's turns could matter, so use a small transformer instead of a mean.
- **Implementation (`--seq`):** a small transformer layer over the agent's turn vectors.
- **Numbers:** AEGIS test pair collapses to 19 to 22. Who&When single-encoder agent F1 is high (50 to 53) but pair is not better. `seqsup5` AEGIS about 22. Dead as a replacement.
- **Source:** the standard transformer (Vaswani et al., 2017) applied by us.

### 4.11 Framework-adversarial training (gradient reversal)
- **Idea:** the five AEGIS frameworks (agentverse, dylan, llm_debate, macnet and so on) have their own style. If the encoder vector hides which framework it came from, it may generalise better to Who&When.
- **Implementation (`--adv-fw LAMBDA`):** a framework classifier on the pooled agent vector, connected through a gradient-reversal layer so the encoder is pushed to remove framework information. Variants: 0.5, 1.0, and combined with contrastive (`advsup`).
- **Numbers:** AEGIS test pair collapses to 19 to 20, Who&When 3.7 to 5.6. Hurts both. Dead.
- **Source:** gradient reversal from Ganin and Lempitsky, 2015, and the DANN paper (Ganin et al., 2016). Applying it to framework identity here is our adaptation.

### 4.12 No-scalars, no-names and no-metadata encoders
- **Idea:** remove possible shortcuts (extra scalar features, agent names, metadata) before fine-tuning.
- **Numbers:** no gain over `es_s0`. Examples: `noscal` AEGIS 29.85 base, Who&When 4.32 / 39.85 / 13.47. `nonames512`: AEGIS 36.08, Who&When 4.44 / 39.59 / 12.00.
- **Source:** our ablations.

### 4.13 Hierarchical mode heads
- **Idea:** the 14 modes fall into three groups (FM-1.x, FM-2.x, FM-3.x). Predict the group first, then multiply the mode probability by its group probability.
- **Numbers (`ww_complex.py`, `es_s0` to `es_s3`):** AEGIS 31.95 / 75.64 / 37.77, Who&When 4.37 / 42.24 / 10.65. Pair on AEGIS rose but agent and Who&When fell. Dead.
- **Source:** hierarchical classification is classical machine learning. Using the MAST groups is our application.

### 4.14 kNN and kNN plus MLP mix
- **Idea:** a non-parametric method that copies labels from similar training agents, so it cannot be overconfident about new names.
- **Numbers:** kNN alone: AEGIS 28.67 / 78.81 / 41.04, Who&When 2.69 / 37.88 / 11.34 (floor level). 50/50 mix with the MLP (`knnmix`): AEGIS 31.07 / 82.87 / 41.77, Who&When 5.09 / 42.01 / 12.67. Better than flat on AEGIS but not on Who&When, and not better than block-dropout.
- **Source:** classical k-nearest neighbours. Our own use.

### 4.15 One-faulty-agent-per-record decode
- **Idea:** Who&When has exactly one faulty agent per record. Keep only the agent with the highest probability.
- **Numbers:** about +0.7 pair without shift, nothing with shift, agent F1 down 4 to 8. Dead.
- **Source:** our own idea, using a known property of the Who&When benchmark.

### 4.16 Test-time adaptation (self-training on unlabeled Who&When)
- **Idea:** use the model's confident predictions on Who&When as extra training data (pseudo-labels), retrain, and predict again, so the model adapts to the new domain.
- **Implementation (`--tta` in `ww_block.py`):** for each Who&When record, the top-agent row is used if its faulty probability is at least 0.5, with modes at or above 0.4. These rows are added to the training set under all four masks, the model is retrained, and results are written as separate `tta_*` rows.
- **Numbers (all blocks on, no shift):**
  - `es_s0` to `es_s6`: pair 6.50 to 7.98 (AEGIS 33.68 to 34.41).
  - `es_s7` to `es_s13`, 1 seed: 4.73 to 7.21. 3 seeds: 5.24 to 6.55.
  - Contrastive ensemble (3 seeds): 5.51 to 6.14, AEGIS unchanged at 36.85.
- **Verdict:** about +1 to +1.5 pair without shift on plain ensembles, never better than label-shift, and stacking on shift hurts (7.76 down to 5.54 on one run). Closed.
- **Source:** pseudo-labelling or self-training (Lee, 2013, "Pseudo-label") and the test-time-adaptation family (for example Wang et al., 2021, "Tent"). This version is our own simple implementation.

### 4.17 Leave-one-framework-out validation
- **Idea:** a Who&When-free way to decide which input blocks to use on a new domain, by holding out each AEGIS framework as a stand-in for "unseen domain".
- **Implementation (`ww_lofo.py`):** train block-dropout on five frameworks, tune thresholds on their validation, score the held-out framework under each of the four masks.
- **Numbers (mean pair / agent / error over 6 frameworks):** names+peer 18.16 / 70.56 / 26.75, names 17.44 / 69.76 / 26.75, peer 17.74 / 62.90 / 27.57, neither 17.62 / 63.42 / 29.89.
- **Verdict:** all blocks on is best or tied, so no principled reason to pick the post-hoc names-only mask for Who&When.
- **Source:** leave-one-domain-out is a standard domain-generalisation evaluation protocol. This implementation is ours.

### 4.18 LoRA language-model reader (Qwen2.5)
- **Idea:** let a small LLM read the whole conversation and answer with the faulty agents and modes, as the Aegis-SFT paper does, but cheaply.
- **Implementation (`aegis/scripts/kaggle_lora_reader.py`):** prompt = system instruction listing the 14 modes, the task, the framework, and the conversation (truncated per turn). Output = one line per faulty agent, for example `Solver: FM-1.2, FM-3.2`. LoRA (rank 16 or 32) on all attention and MLP projections, loss on answer tokens only, trained on AEGIS grouped train, then generated once on AEGIS test and Who&When (zero-shot). Run on a Kaggle T4.
- **Numbers (pair / agent / error micro F1):**
  - Qwen2.5-1.5B, 1,600 records: AEGIS pair 8.22, Who&When 3.10 / 44.25 / 9.09.
  - Qwen2.5-3B, 1,200 records: AEGIS 7.69 / 77.42 / 16.00, Who&When 5.91 / 45.93 / 10.42.
  - Qwen2.5-1.5B, 3,500 records, rank 32: AEGIS 10.36 / 78.91 / 18.84, Who&When 3.23 / 41.94 / 13.48.
- **Verdict:** not competitive with the encoder ensembles. Compute is much smaller than the paper's 14B full fine-tune.
- **Source:** LoRA (Hu et al., 2021). Qwen2.5 models from the Qwen team (2024). The task format and the idea of supervised fine-tuning on AEGIS are from the Aegis paper (Aegis-SFT). Our prompt format and run are our own implementation.

---

## 5. Novelty summary

- **Taken directly from a paper (we reuse the idea or the data):**
  - Dataset, splits, metrics, 14 error modes and the DCL and Aegis-SFT baselines: Kong et al., Aegis-Bench (ICLR 2026) and Cemri et al., MAST.
  - Who&When benchmark: Zhang et al., ICML 2025.
  - MiniLM encoder (Wang et al., 2020) and sentence-transformers (Reimers and Gurevych, 2019).
  - Label-shift EM prior estimate: Saerens et al., 2002.
  - LoRA: Hu et al., 2021.
- **Adapted from a paper (known technique, our own application to this task):**
  - Dropout of name block (Srivastava et al., 2014).
  - Block-dropout over feature blocks (related to ModDrop, Neverova et al., 2016).
  - Gated attention pooling over turns (Ilse et al., 2018).
  - Supervised contrastive loss with error-mode overlap (Khosla et al., 2020).
  - Gradient-reversal framework removal (Ganin and Lempitsky, 2015).
  - Transformer pooling (Vaswani et al., 2017).
  - Self-training (Lee, 2013).
- **Our own design (no direct paper source):**
  - Peer-contrast features (agent versus other agents and versus the query).
  - The four-mask block-dropout recipe that lets one model serve AEGIS and Who&When.
  - The grouped (query-disjoint) split and the two-threshold decoder tuned only on validation.
  - Leave-one-framework-out mask selection protocol.
  - The one-agent-per-record decode and the hierarchical-head variant.
- **Honest note on "novel":** none of these is a fundamentally new learning algorithm. The contribution is the combination (peer-contrast plus name-dropout plus block-dropout plus label-shift plus ensembling) and the careful out-of-domain protocol. Claim "a new method" only for the combination, and call each ingredient by its source.

---

## 6. Did we beat the published numbers?

- **AEGIS-Bench: yes, against Aegis-SFT on all three micro metrics.**
  - Pair: 33 to 37 against 16.62.
  - Agent: about 79 to 82 against 76.53, but the trivial floor is already 68.58, so this margin means little.
  - Error: 41 to 44 against 27.53.
  - Caveats: we used our grouped split (the paper uses the released split), and decoder thresholds were tuned on validation.
- **Who&When:**
  - Pair: 8.24 against o3 7.41 and Aegis-SFT 4.03. This is a beat, but within seed noise (the two halves gave 8.79 and 7.76).
  - Error: 17.12 against 14.88 and 9.87. Beat.
  - Agent: 47.72 against 53.10 and 51.14. Not beaten.
- **Both at once:** one model (`wbes14`) gives AEGIS 32.96 and Who&When 8.24 / 47.72 / 17.12, which beats the paper on AEGIS and on Who&When pair and error. The best-AEGIS model (36.85) reaches Who&When pair only about 5.5.
- **Caveats to state in the thesis:**
  - Who&When has only 184 records, so differences under about 2 pair points are noise.
  - Label-shift is transductive (no labels, but it reads the unlabeled test inputs).
  - The names-only mask rows (9.07, 9.60) were chosen after seeing Who&When, so they are post hoc.
  - The dataset switch from our old self-generated data has not yet been approved by the project guide.

---

## 7. Where the code and logs are

- **Encoder fine-tuning:** `aegis/scripts/kaggle_finetune_minilm.py` (flags: `--attn`, `--seq`, `--adv-fw`, `--supcon`, `--no-scalars`).
- **CPU experiments:** `aegis/src/eval/improve_cpu.py` (peer, name-dropout), `ww_focus.py` (ensemble, label-shift), `ww_block.py` (block-dropout, top-1 decode, test-time adaptation), `ww_complex.py` (hierarchical heads, kNN), `ww_attn.py` (attention pooling), `ww_lofo.py` (leave-one-framework-out), `score_lora.py` (LoRA scoring).
- **LoRA reader:** `aegis/scripts/kaggle_lora_reader.py`.
- **Logs:** raw logs in `aegis/results/runs/` and Kaggle outputs in `aegis/results/kaggle_logs/`. Every run is also a row in `aegis/results/metrics_history.jsonl`.
- **Notes:** `local-notes/results.md` (all numbers, with a full-values appendix), `timeline.md`, `ideas.md` (brainstorm rounds and related papers), `context.md`, `file.md`.
