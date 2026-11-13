# DCL explained: the paper's small model, and how ours differs

This file explains the **DCL** model from the AEGIS paper in simple English: what problem it solves, how it is built, how it is
trained, what the paper leaves unsaid, and how our reconstruction ("DCL-style") differs. Read `context.md` first for the task and
the vocabulary. The rest of the project record is in `current-progress.md`.

**Source:** Kong et al., *Aegis: Automated Error Generation and Attribution for Multi-Agent Systems*, ICLR 2026, arXiv 2509.14295
(version 6). DCL is described in Sec. 5.3 (method) and Sec. 6 (training details), with the losses in Appendix C.3, the training
setup in Appendix D.1, hyper-parameters in Table 7, model sizes in Table 5, results and ablations in Table 1 and Tables 8-10.

**How this file was checked:** the text of the paper was read through a summarising tool on 2026-11-04, not by reading the PDF page by
page. Things the paper states are marked as such. Things the text did not state are listed in section 7 as "not given"; some of them
may be in figures or tables that the text extraction missed. Nothing in this file about the paper's internals is guessed.

---

## 1. The problem DCL is meant to solve

A multi-agent system is a team of language-model agents. When the team fails, we want to know **which agents were faulty and what
kind of mistake each made**. The paper's data (AEGIS-Bench) gives, for each conversation, the list of turns and the true list of
`(agent, error mode)` pairs, with 14 error modes.

The paper compares three kinds of approaches:
1. Prompting a big language model with no training (zero-shot), like our Qwen reproduction.
2. Fine-tuning a language model on the data (Aegis-SFT, Aegis-GRPO).
3. A **small, cheap, specially designed model** trained from scratch on the data. That is DCL.

DCL is the third kind. It is the paper's answer to "can a very small model do this, if the model is designed well?". It is also
the model our graph network is meant to be compared with, which is why it matters for the thesis.

## 2. The name

**DCL = Disentangled Contrastive Learning.**
- *Contrastive learning* means training the model by showing it examples that should look similar (positives) and examples that
  should look different (negatives), and rewarding it for pulling the first together and pushing the second apart.
- *Disentangled* means the model keeps two separate "views" of a fault, one for **which agent** and one for **which error type**, and
  only combines them at the end, instead of learning one mixed-up answer.

## 3. How DCL is built, step by step

Think of the model as a detective reading a transcript, with a list of suspects and a list of crime types.

**Step 1: treat the conversation as a bag of turns.** The conversation is split into its turns. The order is ignored. There are
no arrows between turns. The paper says DCL "represents each trajectory as a bag of turns".

**Step 2: turn each turn into numbers (the encoder).** Every turn goes through a small pretrained language model called
**all-MiniLM-L6-v2**. It outputs 384 numbers per turn, which capture the turn's meaning. A projection layer then shrinks these to
128 numbers. The same encoder is used for every turn ("shared encoder"). In the paper, **the encoder is trained together with the
rest of the model** (end to end), so it can adapt to this task.

**Step 3: find the suspicious turns (attention).** Not every turn matters. Most turns are normal. A fault usually shows up in
one or two turns. The model uses **Multiple Instance Learning attention** (a method from Ilse et al., 2018) to give each turn a
weight, a score for "how much evidence of a fault is in this turn". Then it keeps only the **top 3** turns by weight (K = 3, chosen
on validation data). These are the "evidence turns".

*Why this design:* in a long conversation the label belongs to a whole agent or whole conversation, but the evidence is in a few
turns. This lets the model learn where to look without anyone marking the faulty turns.

**Step 4: two banks of reference points (prototypes).** The model keeps two lists of reference vectors:
- one **agent bank**: a reference vector for each agent role,
- one **error bank**: a reference vector for each of the 14 error modes.

The paper says these are initialised from **textual definitions** (descriptions written in words), then adjusted in training. Think
of them as "what a typical Evaluator looks like" and "what a typical FM-2.5 fault looks like".

**Step 5: compare evidence turns with the reference points.** Each evidence turn is compared with every agent reference and every
error reference using **cosine similarity** (a number saying how close two vectors point in the same direction). Close to an agent
reference means "this turn looks like it came from that kind of agent"; close to an error reference means "this turn looks like that
kind of fault". These similarities become two probability tables:
- **p^A**: how likely is each agent to be the faulty one,
- **p^E**: how likely is each error mode to be present.

This is the "disentangled" part: agent and error are scored separately.

**Step 6: combine into pair probabilities.** For each possible `(agent, error)` pair, a small **gate** combines the agent probability
and the error probability into a pair probability **p^P**. The paper's algorithm describes the gate as "product / bilinear". A product
means multiplying (a pair is likely only if both its agent and its error are likely). A bilinear head is a learned version of that
(Table 7 gives its rank as 64). The final answer is the set of pairs whose probability is high enough.

## 4. How DCL is trained

**The three loss terms.** A loss is a number saying how wrong the model currently is; training changes the weights to make it
smaller. DCL adds three losses, with weights from Table 7:

`total = 1.0 × L_cls + 0.4 × L_con + 0.6 × L_hier`

1. **L_cls, classification loss (weight 1.0).** The ordinary "did you get the labels right" loss, applied at three levels at once:
   agent, error mode and pair. The paper's text describes standard binary cross-entropy (BCE), but Table 7 says *asymmetric* BCE
   (ASL), which treats the many "not faulty" labels more gently than the few "faulty" ones. The paper does not say which is used; the
   two descriptions disagree.
2. **L_con, contrastive loss (weight 0.4, temperature 0.07).** This is the "contrastive" part. For each evidence turn:
   - **Positives** (should be close): the turn paired with the true agent reference, and with the true error reference. The paper
     also says the main positive is the matching "clean" turn from the **original, deterministic successful run**. The paper's wording
     implies that each failed conversation was made by injecting a fault into a run that had succeeded, so a clean twin of the turn
     exists; I have not checked this against the released data.
   - **Negatives** (should be far): the turn paired with wrong references, plus **hard negatives**, namely evidence turns taken from
     *other* conversations paired with their own correct references.

   Intuition: the model learns what a faulty turn looks like by comparing it with its clean twin and with wrong answers.
3. **L_hier, hierarchy consistency (weight 0.6).** A common-sense rule: a pair cannot be more likely than its own agent or its own
   error. If the pair `(Solver, FM-3.2)` has probability 0.9 but the error FM-3.2 only has 0.3, that is inconsistent, and the
   model is penalised (squared hinge on `p^P − min(p^A, p^E)`). The paper's ablation shows this matters a great deal: without it,
   pair-level accuracy nearly collapses.

**Training settings (Table 7 and Sec. 6).**

| Setting | Value |
|---|---|
| Encoder | all-MiniLM-L6-v2, shared, trained end to end |
| Optimizer | AdamW, weight decay 0.01 |
| Learning rate | 1e-4, cosine schedule with 10% warm-up |
| Batch size | 128 |
| Length | 2 epochs of 500 steps (1,000 steps in total) |
| Evidence turns K | 3 |
| Contrastive temperature | 0.07 |
| Encoder and projection size | 384 and 128 |
| Gradient clipping | 1.0 |
| Precision and hardware | bf16, 4 NVIDIA 4090 GPUs |
| Model size | about 23 to 35 million parameters (Table 5) |

## 5. The published results

On the AEGIS-Bench test set and on Who&When (micro-F1 / macro-F1, percent), from the paper's Table 1:

| | Pair | Agent | Error |
|---|---|---|---|
| AEGIS-Bench | 8.33 / 5.30 | 22.93 / 20.23 | 24.73 / 27.70 |
| Who&When | 1.60 / 0.77 | 8.40 / 6.07 | 14.67 / 10.57 |

The average of these 12 numbers is **12.61**, against 4.08 for the paper's random baseline. The paper's ablations (average scores):
only-mix head 12.42, only-bilinear head 10.01, without "intent" 10.16, without the consistency loss 9.52. What the "intent" and "mix head"
components are is not explained in the text I read.

**An important warning about these numbers.** On our scoring code, a rule with no model at all ("predict every agent that appears, with
all 14 modes") already gets about 70 on agent-level micro-F1 and about 28 on error-level micro-F1. The paper's DCL agent score (22.9) and
error score (24.7) are **below** that rule. So either the paper's agent-level numbers are measured in a way we have not reproduced, or the
model predicts agent names in a way that does not match the labels. We could not resolve this. It is why this project always reports the
trivial floor next to any comparison with DCL.

## 6. What our "DCL-style" version does differently

There is no released DCL code, so `aegis/src/baselines/dcl_style.py` is our own reconstruction. It follows the paper's description as
far as it is given, but it is **not DCL** and should always be called "DCL-style".

| Part | Paper's DCL | Our DCL-style |
|---|---|---|
| Encoder | MiniLM, fine-tuned end to end | MiniLM **frozen**; vectors read from the saved cache |
| Trainable size | about 23 to 35 million parameters | about 0.2 million parameters |
| Learning rate and length | 1e-4, 1,000 steps, cosine schedule | 1e-3, up to 20 passes with early stopping on validation loss |
| Batch size | 128 | 64 |
| Contrastive loss | turns vs prototypes, clean twin as positive, hard negatives from other conversations | agents that share an error mode are pulled together (our own simplification); the clean twin is **not used** |
| Agent references | textual definitions of roles | the agent's name, embedded as text |
| Pair head | product / bilinear gate | bilinear term of rank 64 added to the mode scores |
| Evidence pooling | MIL attention weights, top 3 | average of the top 3 turn scores per agent |
| Final decision | not specified in the text read | one threshold tuned on validation, at least one pair per conversation |

The same loss weights (1.0, 0.4, 0.6), temperature (0.07), K = 3 and projection size (128) are used.

**Where this leaves us.** Our version scores *higher* than the published DCL on 10 of the 12 micro-F1 numbers (for example AEGIS-Bench
pair 15.1 against 8.3 when trained on the released split). That does not mean our model is better. It means our version is not the same
model, and the published agent-level numbers sit under the trivial floor. On our own validation set the DCL-style model is also worse
than a plain flat neural network (pair 13.5 against 23.1), so the prototype and top-3 design did not help in this frozen form.

## 7. What the paper does not say (in the text read)

- how the prototype banks are initialised beyond "textual definitions", and how many prototypes there are,
- how the similarities are turned into the probabilities p^A and p^E,
- how a failed turn is matched to its clean twin in the original successful run,
- what "intent" means in the "w/o intent" ablation, and what the "mix head" is,
- how the gate strength (1.0 in Table 7) is used,
- whether the loss is plain BCE or asymmetric BCE (the text and Table 7 disagree),
- how the final list of pairs is chosen from the probabilities (threshold, top-k, or something else),
- the per-GPU batch split.

## 8. What this means for the thesis

1. Compare with DCL only together with the trivial floor, the frequency baseline and the flat MLP, and state that the published DCL could
   not be reproduced because no code was released.
2. Call our model "DCL-style (frozen encoder)" every time, and list the differences in section 6.
3. Base claims on the grouped split and on pair-level and macro scores, where models actually differ.
4. If the clean twin of each turn is present in the released data (not yet checked), the paper's full contrastive loss could be tried. That
   would be a closer reproduction, but it needs a different data loader, because our loader reads only a safe list of fields.
