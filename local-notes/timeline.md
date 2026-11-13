# Timeline: what we try, in order, and what each approach means in simple words

LOCAL ONLY (`local-notes/`, git-excluded). Never commit or push. Companion: `results.md` (the numbers, filled in as jobs finish), `ideas.md` (all ideas), `loop.md` (how the loop works), `file.md` / `context.md` (detailed running notes).
Format rule from the user: bullet points, no tables.
Window: 3 days from 2026-11-12 (evening). Goal: improve scores on AEGIS-Bench AND Who&When. Rule: only small jobs first (about 0.5 to 1 hour on one T4 GPU each); every graph idea runs with its 0-layer control; 5 seeds before believing a 1 to 3 point difference.
Setup: three Kaggle accounts (tokens live in `kaggle.txt`, never copied into notes). Account A = aryanmahajan665, B = aryanmahajan7, C = amahaj55. The two input datasets (raw data, grouped ids) were copied as private datasets into A and C.

## Starting point (what we must beat)
Grouped test, micro pair / agent / error: best so far = name-embedding flat MLP 29.41 / 79.78 / 39.09 (average of 2 clean encoders). Who&When best = 5.04 / 40.85 / 15.04. Floors: AEGIS test 7.35 / 68.58 / 27.14, Who&When 3.42 / 38.32 / 13.33.
Diagnostic (2026-11-12): the weak spot is picking the right MODE, not finding the faulty agent (knowing the agents would add about 3 pair points; knowing the modes would add about 28).

## Wave 1 (started 2026-11-12 evening): three GPUs, six small jobs at the same time
Each job fine-tunes a small text encoder on the grouped training data (about 15 to 40 minutes on a T4), then exports the fine-tuned message vectors. Afterwards the CPU models (flat MLP with name embedding, GNN) are re-run on those vectors and scored once on the test set and Who&When.

- aegis-es-s2 and aegis-es-s3 (account A): encoder seeds 2 and 3.
  - Simple words: the same recipe as before with a different random start.
  - Why: our scores move 1 to 3 points just from the seed. With 4 to 5 encoder seeds we can average them (ensemble) and see which improvements are real.
- aegis-es-s4 (account B): encoder seed 4.
  - Simple words: the same recipe, a fifth random start.
  - Why: completes five seeds.
- aegis-len512 (account B): reads 512 tokens per message instead of 256.
  - Simple words: right now every message is cut after 256 tokens, so the end of long messages is never seen.
  - Why: faults may hide in the long tail of a message.
- aegis-l12 (account C): encoder MiniLM-L12 (12 layers instead of 6).
  - Simple words: a somewhat bigger brain of the same family.
  - Why: a cheap test of whether a deeper encoder reads the faults better; also a different model to ensemble with.
- aegis-bgesmall (account C): encoder bge-small-en-v1.5.
  - Simple words: a different small encoder family trained with a different recipe.
  - Why: diversity for ensembling, and a chance of better text understanding at the same speed.

## Wave 1 CPU work (I do these myself while the GPUs run; one heavy local job at a time)
Run on the already-saved fine-tuned vectors (clean encoders es_s0, es_s1), compared with the same baseline and 5 seeds.
1. Per-class thresholds. Simple words: today one cut-off decides "is this mode present" for every mode. We give each mode its own cut-off, tuned on validation only.
2. Attention pooling. Simple words: today an agent is summarised by the plain average of its messages, so one suspicious message gets drowned. Attention learns which messages to listen to.
3. Hard pair rule. Simple words: build into the model that a pair can never be more likely than its agent or its mode, and train on pairs directly.
4. Peer contrast. Simple words: for each agent, add how different its text is from the other agents' average and from the question.
5. Label-shift correction for Who&When. Simple words: Who&When has a different mix of error types than AEGIS; estimate the mix from our own predictions (no labels) and re-weight.
6. Name shuffling. Simple words: during training, sometimes hide or swap the agent name so the model cannot rely on names that do not exist in Who&When.

## Later waves (only if wave 1 shows something, or time remains)
- Sibling pseudo-clean reference and the sibling graph (compare each agent with the same task's other copies; the best novelty bet).
- Content-flow edges and de-propagation (separate the originator of a fault from the agents that only repeated it).
- Leave-one-framework-out validation (honest way to pick the Who&When recipe).
- Ensembling the best encoders (needs the seeds above).
- Optional and big: LoRA small LLM; zero-shot repeat for the 27 versus 49 agent-score puzzle.

## Log of what actually happened (newest last)
- 2026-11-12 evening: read `kaggle.txt` (three tokens, three accounts), copied the two input datasets into accounts A and C (private), added `--max-len` to `aegis/scripts/kaggle_finetune_minilm.py`, launched the six wave-1 kernels (see table). Diagnostic script `aegis/src/eval/diagnostic.py` finished earlier (mode identification is the bottleneck).

- 2026-11-12 20:16: first CPU results landed (improve_cpu es_s0): peer-contrast features +3.8 test pair; name-dropout +6.0 Who&When agent. Now testing the combination and replicating on es_s1.

- 2026-11-12 20:23: encoder seeds 2 and 4 finished (losses match seed 0); two more seeds (5, 6) launched on accounts A and B. Downstream scoring of the new vectors queued behind the CPU replication run.

- 2026-11-12 20:40: three more encoders finished (es_s3, len512, bgesmall); replication on es_s1 confirms peer-contrast (+4.4) and name-dropout Who&W agent (+4.2); full per-seed appendix added to the notes at the user request.

- 2026-11-12 20:50: reading 512 tokens per message (len512) raised in-domain test pair from about 28 to about 34 (36 with peer features). New job aegis-chunk: instead of cutting long messages, split them into pieces and average the pieces, so the whole message is read. Simple words: the encoder used to throw away the end of long messages; now it reads (up to 3 pieces of) the whole message.

## TARGET CHANGE (2026-11-12 22:10, from the user)
- New priority: improve Who&When as much as possible, using the paper-side approaches and our novelties, WITHOUT hurting AEGIS-Bench. AEGIS gains so far (peer-contrast, longer messages) are kept as the in-domain base; new experiments are judged first on Who&When pair / agent / error and must keep AEGIS test pair within noise of the current best.
- Who&When-focused experiments, simple words:
  - Name-dropout (already helps agent): the model is trained both with and without agent names, so it does not depend on names that do not exist in Who&When.
  - Ensembling across encoders: average the predictions of many encoders; Who&When has only 184 records, so noise is large and averaging should steady it.
  - Label-shift correction: Who&When has many more FM-2.6 and FM-3.3 errors than AEGIS; estimate that mix from our own predictions (no labels) and re-weight, so the model expects the right mix.
  - Next ideas queued: attention pooling (find the one suspicious turn), leave-one-framework-out validation (choose the recipe without looking at Who&When), name shuffling variants, a role-mapping for unseen agent names.
- 2026-11-12 22:10: encoder gains recorded (see results.md); ww_focus.py written (ensemble + name-dropout + label-shift) and queued on the seven 256-token encoders.

- 2026-11-12 22:25: user decided to skip the LoRA language-model side track for now. Remaining Who&W plan stays on the cheap encoder/CPU ideas.

## Attention pooling, two versions (2026-11-12 22:58)
- Simple words: today an agent's messages are averaged into one vector. Attention pooling lets the model give more weight to the messages that look faulty, because a fault usually sits in one or two messages and the average dilutes it. Aimed at the Who&When agent level (we are at about 41 to 45, the paper's best is 51 to 53).
- CPU version (`aegis/src/eval/ww_attn.py`): learns the attention weights on top of saved encoder vectors, trained with name-dropout, compared against a plain-mean control with the same recipe. One-seed smoke on es_s0 (wiring check only): test 27.65 / 82.00 / 38.88, Who&W 7.04 / 42.20 / 16.80. Three-seed runs queued (es_s0, then len512_s1).
- GPU version (kernel `aryanomgh65/aegis-attn`, account D, the 4th GPU): the encoder itself is fine-tuned with attention pooling over turns (`--attn` in `kaggle_finetune_minilm.py`), so it learns turn vectors that suit attention. Needs a GPU because it fine-tunes the language-model encoder. Same recipe as the 256-token clean encoders (seed 0, 15% early-stop slice), so it compares directly with es_s0.
- Datasets for account D created as private copies (aegis-raw-data, aegis-grouped-ids).

- 2026-11-12 23:00 USER: entire focus on Who&When; once the old jobs and GPUs free up, only Who&When experiments run (no more AEGIS-only encoder seeds). Queue of freed-GPU Who&W jobs, in order: (1) name-shuffle encoder (agent names swapped between agents during fine-tune), (2) leave-one-framework-out encoders (6 held-out runs) for honest recipe choice, (3) more seeds of whichever pooling (mean/attn) wins on Who&W.

## Escalation ladder (agreed 2026-11-12 23:10)
- Trigger: after the Who&W ensemble job, attention pooling (CPU and GPU) and name-shuffling are done, if Who&W agent is still below about 47 and pair below about 7.5, escalate.
- Rungs: (1) sibling counterfactual-pair training (lead A in ideas.md section 7), (2) deviation then de-propagation, (3) small language-model reader on Who&W, (4) content-flow graph edges. Cheap CPU leads in parallel: scalar-dropout (lead B), test-time adaptation as a separate transductive row (lead D).
- Standing rule: brainstorm research papers again every few hours (new search round appended to ideas.md) and consider switching to a stronger model for the hard design steps.

- 2026-11-12 23:20: l12len512 seeds 1 and 2 and len512 seed 2 finished and saved; L12+512 is the lowest-loss recipe on 3 seeds (0.1335/0.1358/0.1369). Scoring queued (chain8) behind ww_focus and chain7. A and B and C are now idle; per the Who&W-only rule they get Who&W jobs next.

- 2026-11-12 23:27: Who&W GPU jobs launched. nonames = encoder fine-tuned without seeing agent names or framework in the head, so it cannot lean on names and must put the signal into the message vectors. nometa = same plus the count scalars (turn counts, position) removed, a targeted-augmentation test (ideas.md lead B). Both compared against es_s0 (same seed and recipe).

- 2026-11-12 23:12: FIRST BIG WHO&W RESULT. ww_focus on 7 encoders: nd ensemble Who&W 6.80/45.78/16.12; nd + label-shift 8.46/46.42/17.08 (pair above paper o3 7.41, error above o3 14.88, agent below 51-54; macro error drops to 6.0). Trade-off: nd has AEGIS test pair 26.9 (peer features restore 33.9 but Who&W falls to 5.4). Next: block-dropout (train a model that keeps AEGIS peer gain and works without peer block), CPU leave-one-framework-out on frozen vectors to decide, attention pooling (chain7). attn encoder loss no better than es_s0. Launched nonames seed 1 on account D.

- 2026-11-12 23:14: attention pooling (CPU, es_s0) result: no Who&W gain (pair 6.40 vs 6.04 control, agent 44.99 vs 47.04). Dropped as a lever. Control reproduced earlier namedrop row exactly. Escalation trigger check: best Who&W agent still about 47 and pair 6.8 (8.46 with label-shift), so next rungs are block-dropout and the sibling counterfactual idea.

- 2026-11-12 23:35: data check killed the literal sibling-pair idea (siblings share only 7.5% of text; no clean records). New cheap Who&W idea from the data: AEGIS has 2-4 faulty agents per record but Who&W exactly 1, so train/decode with a per-record fault-count prior (ideas.md section 8). Complex rung now = small LM reader and de-propagation; GPUs get these as they free up.

- 2026-11-12 23:45: nonames/nometa encoders saved; downstream Who&W scoring queued. These test whether an encoder fine-tuned without name/count information transfers better to Who&W than es_s0.

- 2026-11-12 23:32: attention pooling is AEGIS-positive (+2.6 to +2.9 pair, tighter spread) and Who&W-neutral on two encoders; keep as a recipe component. len512_s1 peernd Who&W 7.09 did not replicate on len512_s2 (4.36).

## Complex ideas round (user approved 2026-11-12 23:55: try these first, then LoRA)
- Order agreed: (1) framework-adversarial encoder, (2) contrastive encoder, (3) hierarchical mode heads (CPU), (4) kNN / prototype classifier (CPU), (5) sequence model over turns, (6) test-time adaptation (separate transductive row), (7) de-propagation, then (8) LoRA language-model reader.
- Framework-adversarial encoder (simple words): while fine-tuning, a small side network tries to guess which framework (agentverse, macnet, ...) a message came from, and the encoder is trained to make that guess fail. The vectors keep error information but lose framework style, which should transfer better to Who&When. Kernel `amahaj55/aegis-advfw` (weight 0.5, MiniLM 256, seed 0, same recipe as es_s0).
- Contrastive encoder (simple words): extra training signal that pulls together agents with the same error mode and pushes apart different ones, across frameworks, to sharpen mode identification (the bottleneck). Kernel `aryanomgh65/aegis-supcon` (weight 0.2, same recipe).
- Code: `--adv-fw` and `--supcon` flags in `kaggle_finetune_minilm.py`. Not smoke-tested on GPU (no CUDA here); check logs at COMPLETE.
- 2026-11-12 23:50 USER: explore ALL the complex ideas before LoRA; keep assigning them as GPUs free up; only then move on to LoRA. Queue (GPU, in order as accounts free): framework-adversarial (running C), contrastive (running D), then sequence-model-over-turns encoder, adversarial+contrastive combined, adversarial with other weights, test-time-adaptation / self-training variants (CPU), hierarchical heads + kNN (CPU, ww_complex.py queued), de-propagation (CPU), and finally LoRA language-model reader.

- 2026-11-12 23:58 USER: assign complex jobs IMMEDIATELY as GPUs free, then LoRA once all are done. Added --seq (order-aware small transformer over an agent's turn vectors, end-to-end) to the fine-tune script; helper launch.sh in the scratchpad launches any recipe in one command. Next GPU jobs in order: seq (sequence model), advsup (adversarial + contrastive), advfw weight 1.0 or 0.2 depending on the advfw result, then LoRA.

- 2026-11-13 00:10: LoRA reader prepared and ready to launch when GPUs free: aegis/scripts/kaggle_lora_reader.py (Qwen2.5-1.5B-Instruct, LoRA r16, trains on 1600 AEGIS grouped-train records within a 170-minute cap, generates Who&W first, then test, then 200 val) and aegis/src/eval/score_lora.py (CPU scoring, ledger phase 11). Prompt/answer/parse functions CPU-tested; model code NOT GPU-tested (no CUDA here).
- 2026-11-13 early: advfw+supcon saved, relaunched advsup (D) and advfw1 (C); chain13 queued.
- 2026-11-13: seq saved, chain14 queued, aegis-noscal launched on A.
- 2026-11-13: block-dropout (ww_block) names+peer: AEGIS 32.93, WW 6.10/44.57/12.65, +shift 8.79/48.54/17.86 = new best. advfw1/advsup saved; supcon5, seqsup launched; chain15 queued.
- 2026-11-13: noscal, nonames512 saved; attnsup, l12sup512 launched; chain16 queued.
- 2026-11-13: nonames scored; chain11 on nometa.
- 2026-11-13: nometa scored (no gain); supcon5, seqsup saved; advfw-s1, supcon-s1 launched; chain17 queued.
- 2026-11-13 00:45: attnsup saved, attnadvsup launched, chain18 queued.
- 2026-11-13 ~00:55: len512_s3 scored; chain11 done; chain12 running.
- 2026-11-13: chain12 done (hier, kNN dead for Who&W; knnmix AEGIS ok, WW lower); advfw_s1, supcon_s1 saved; advsup-s1, noscal-s1 launched; chain19 queued.
- 2026-11-13 01:05: attnadvsup saved; LoRA kernel aegis-lora launched on A.
- 2026-11-13 06:05: batch scoring recorded (supcon/supcon5 best Who&W gains; adv dead; seq trade-off); LoRA relaunched after torchao fix; supcon5-s1, supcon10, attnsup5 launched; chain21 running.
- 2026-11-13 06:02: queued chain22 = ww_block (block-dropout) on supcon, supcon5, attnsup after chain21; kernels (lora, supcon5-s1, supcon10, attnsup5) running.
- 2026-11-13 06:16: supcon5_s1, supcon10 saved; supcon5-s2, seqsup5 launched; chain23 queued.
- 2026-11-13 06:20: supcon_s1 does not replicate seed 0 (WW pair 4.2-4.8); attnsup5 saved; supcon10-s1 launched; chain24 queued.
- 2026-11-13 06:35: supcon5_s2 (B) saved clean; scoring queued chain25; launched supcon5-s3 (B, seed 3) to build a multi-seed supcon5 ensemble for block-dropout.
- 2026-11-13 06:37: seqsup5 (C) saved clean; scoring queued chain26; launched supcon5-s4 (C). Fixed chain25 dir-name bug (supcon5_s2 -> supcon5-s2).
- 2026-11-13 06:40: supcon10_s1 (D) saved clean; scoring queued chain27; launched supcon5-s5 (D, seed 5).
- 2026-11-13 06:55: supcon5_s3 (B), supcon5_s4 (C) saved clean; scoring queued chain28; launched aegis-l12sup256 (B, L12 256 supcon 0.5) and aegis-attnsup5-s1 (C).
- 2026-11-13 07:00: supcon5_s5 (D) saved clean; scoring queued chain29; launched supcon3-s6 (D).
- 2026-11-13 07:05: CPU scoring is the bottleneck; started chain22 (ww_block on supcon,supcon5,attnsup) NOW in parallel with chain21 (2 CPU jobs at once, deliberate; user said take any permission and keep going) because it is the decisive experiment.
- 2026-11-13 07:13: attnsup5_s1 (C) saved clean (scoring to queue); launched supcon5-s7 (C). advsup_s1 scored (adv+supcon seed 1: see improve_cpu_advsup_s1.log).
- 2026-11-13 07:18: supcon3_s6 (D) saved clean (scoring to queue); launched supcon5-s8 (D). Block-dropout on contrastive encoders (chain22): supcon, supcon5 trained, attnsup next.
- 2026-11-13 07:23: l12sup256 (B) saved clean (scoring to queue); launched l12sup512-s1 (B, L12 512 supcon 0.5 seed 1).
- 2026-11-13 07:30: wbsup done (AEGIS 34.87, WW 5.35/47.03/10.04; no new WW best); enc_dir fixed; wbsc5 and wbmix block-dropout ensembles started (chain30).
- 2026-11-13 07:33: supcon5_s7 (C) saved clean; launched l12sup256-s1 (C).
- 2026-11-13 07:37: supcon5_s8 (D) saved clean; launched len512sup (D, L6 512 tokens supcon 0.5).
- 2026-11-13 07:45: LoRA scored: AEGIS test 8.22/74.92/19.59, WW 3.10/44.25/9.09 -> not competitive; A relaunched with seqsup5-s1.
- 2026-11-13 07:48: chain21 done and recorded (l12sup512 strong AEGIS/weak WW; noscal_s1 no gain; adversarial dead x3).
- 2026-11-13 07:55: supcon5_s1 scored: WW pair 4-5 (seed 0 gain not replicated); waiting for wbsc5/wbmix.
- 2026-11-13 08:02: seqsup5_s1 (A), l12sup256_s1 (C) saved clean (scoring to queue); launched plain-recipe es-s7 (A) and es-s8 (C) to enlarge the es block-dropout ensemble (current best WW row).
- 2026-11-13 08:11: len512sup (D) saved clean (scoring to queue); launched plain es-s9 (D).
- 2026-11-13 08:15: supcon10 scored (no gain, contrastive weight not a lever); l12sup512_s1 saved; es-s10 launched on B.
- 2026-11-13 08:20: es_s7 (A), es_s8 (C) saved clean; launched es-s11 (A), es-s12 (C). wbsc5 trained all 6 encoders; summary pending.
- 2026-11-13 08:25: wbsc5 done: AEGIS test 36.15, WW 6.01/47.39/10.07 (shift hurts); best WW row still es block-dropout+shift 8.79/48.54/17.86; wbmix next.
- 2026-11-13 08:30: es_s9 (D) saved clean; launched es-s13 (D).
- 2026-11-13 08:35: user: stop launching new GPU jobs, just complete running ones (es-s10..s13); CPU scoring/ensembles continue.
- 2026-11-13 08:35: es_s10 (B) saved clean; B left idle per user. attnsup5 scored (see improve_cpu_attnsup5.log). Not launching new GPU jobs.
- 2026-11-13 08:40: es_s11 (A), es_s12 (C) saved clean; A, B, C idle per user; only es-s13 (D) still running. Final block-dropout ensemble with es_s7..s13 to queue when es_s13 lands.
- 2026-11-13 08:50: es_s13 (D) saved clean; ALL GPU jobs finished and saved, all 4 GPUs idle (per user). Queued chain31 = ww_block on es_s0..es_s13 (14 plain-recipe encoders, tag wbes14) after wbmix.
- 2026-11-13 08:52: supcon5_s2 scored (see improve_cpu_supcon5_s2.log); wbmix 7/10 trained.
- 2026-11-13 09:16: wbmix done: all-on AEGIS 35.45, WW 5.53/46.16/10.85 (+shift 7.32/47.01/16.76); names+shift 9.07/47.79/18.63 (post-hoc, AEGIS 31.23). No clear new winner; wbes14 next.
- 2026-11-13 09:21: seqsup5 scored: AEGIS pair ~22 (-10), WW no gain: sequence encoder dead. wbes14 training 1/14.
- 2026-11-13 09:40: user allows GPUs again; launched aegis-lora-3b (B) and aegis-lora-long (C). A, D idle.
- 2026-11-13 09:40: supcon10_s1 scored (see improve_cpu_supcon10_s1.log); wbes14 5/14 trained; lora-3b, lora-long running.
- 2026-11-13 09:57: CPU-offload to Kaggle: dataset aegis-cpujob uploaded to A (code + data + es_s7..s13 vectors); pushed kernel aegis-cpu-es7 (A; ww_block wbes7b, output ww_block_wbes7b.log + metrics_history_wbes7b.jsonl; save to aegis/results/kaggle_logs/cpu_es7). D dataset upload in progress, then push aegis-cpu-scb (wbscb). Staging dir D:/aryan/cpujob_stage (delete after). C: drive full (615MB free), told user.
- 2026-11-13 10:02: aegis-cpu-scb (D) first push failed (dataset not attached yet; log saved in kaggle_logs/cpu_scb_fail); re-pushed v2 with dataset. aegis-cpu-es7 (A) running. wbes14 9/14 trained.
- 2026-11-13 10:20: cpu_es7 (A) saved; wbes7b: es_s7..s13 all-on AEGIS 33.23, WW 5.24/43.03/11.52, +shift 7.76/46.82/17.99 -> best es row reproducible (~8 pair). A idle. wbes14 11/14.
- 2026-11-13 10:25: cpu_scb (D) saved; wbscb: all-on AEGIS 36.85, WW 5.51/46.98/9.99 (replicates wbsc5). A and D idle. wbes14 13/14 trained.
- 2026-11-13 10:32: wbes14 done: all-on AEGIS 32.96, WW 5.56/43.75/11.76; +shift 8.24/47.72/17.12 = converged es-ensemble estimate (halves 8.79 / 7.76). chain28 done. A, D idle; B,C LoRA running.
- 2026-11-13 10:50: chain29 done (all scoring chains finished). New CPU idea launched (chain32, tag wbtop1): per-record fault-count prior for Who&W: one-faulty-agent-per-record decode (top1: modes >= threshold; top1m: single best mode), flag --top1 added to ww_block.py; es_s0..s6 block-dropout, reports it as a separate benchmark-structure-prior row (no Who&W labels used; Who&W 'exactly one faulty agent' is a known dataset property).
- 2026-11-13 11:15: wbtop1 done: one-agent-per-record decode DEAD (pair +0.7 without shift, 0 with shift, agent -4..-8). Reproduces 8.79/48.54/17.86 exactly (deterministic pipeline). CPU free.
- 2026-11-13 11:21: brainstorm round (ideas.md section 9): nothing cheap beyond a content-flow originator feature.
- 2026-11-13 12:00: lora_3b (B) saved+scored: AEGIS 7.69/77.42/16.00, WW 5.91/45.93/10.42 (better than 1.5B LoRA, not competitive with ensembles). B idle; lora-long (C) running.
- 2026-11-13 13:20: lora_long (C) saved+scored: AEGIS 10.36/78.91/18.84, WW 3.23/41.94/13.48. LoRA closed: no run beats encoder ensembles (best LoRA WW = 3B 5.91). All 4 GPUs idle, CPU idle; nothing running.
- 2026-11-13 13:30: CPU idea launched: test-time adaptation / self-training on unlabeled Who&W (--tta in ww_block.py; confident own predictions (top agent p>=0.5, modes >=0.4) become extra training rows, retrain; separate 'tta_*' rows, transductive, no Who&W label). Smoke test on es_s0, 1 seed (ww_block_ttasmoke.log); full run on es_s0..s6 next if it works.
- 2026-11-13 13:40: TTA smoke promising for pair (+1.4, single enc/seed; agent -1.9). Launched full test: local ww_block --tta es_s0..s6 1 seed (ttaA, log ww_block_ttaA.log) and Kaggle A aegis-cpu-tta (es_s7..s13, 1 seed, ttaB; output via --file-pattern 'ttaB|\.log').
- 2026-11-13 13:46: ttaB (Kaggle A) saved: TTA lifts WW pair 4.73->7.21 (all-on, no shift), AEGIS flat; same level as label-shift, not stacking. ttaA (local es_s0..s6) summary pending.
- 2026-11-13 13:51: ttaA done: all-on pair 6.50->7.98 (AEGIS 33.68->34.41); names+tta+shift 9.60/50.45/16.35 but not replicated in ttaB (7.06/47.64/13.93). TTA ~ label-shift equivalent, no new best. Idle: all GPUs, CPU.
- 2026-11-13 14:00: launched leave-one-framework-out validation of block-dropout masks (new aegis/src/eval/ww_lofo.py; es_s0..s2, 1 seed; log ww_lofo_lofo1.log): which mask (names+peer/names/peer/neither) transfers best to an unseen AEGIS framework, a Who&W-free way to pick the Who&W mask. Held-out frameworks with enough rows: agentverse, dylan, llm_debate, macnet (magentic_one/smoagents tiny).
- 2026-11-13 14:11: LOFO done: all-on best/tied on pair (18.2 vs 17.4-17.7), agent +6-7 with names; no mask justified beyond all-on. Post-hoc names-only rows stay post hoc. Next CPU idea: per-record fault-count correction / de-propagation.
- 2026-11-13 14:20: brainstorm round (ideas.md sec 10): nothing new/cheap; fault-count correction == top1 (already dead). Remaining untested low-value: de-propagation, name-to-role mapping. Queue effectively exhausted; all GPUs+CPU idle.
- 2026-11-13 14:40: launched noise-tightening replicates of TTA (3 head seeds each, same code): Kaggle A aegis-cpu-tta3a (es_s7..s13, tag ttaB3) and Kaggle D aegis-cpu-tta3d (es_s0..s6, tag ttaA3). Save with --file-pattern 'tta[AB]3|\.log' into kaggle_logs/cpu_tta3a, cpu_tta3d. B, C idle, local CPU idle.
- 2026-11-13 14:35: aegis-cpu-tta3d (D) failed in 90 s: D's dataset holds only the supcon/attnsup encoders, not es_s0..s6 (FileNotFoundError es_s0; log saved kaggle_logs/cpu_tta3d). Re-pushed v2 with the 7 supcon/attnsup encoders (wbscb set), tag ttaS3: TTA on the best-joint contrastive ensemble. tta3a (A, es_s7..s13 x3 seeds) still running.
- 2026-11-13 14:56: ttaB3 saved+scored: TTA +1.3 pair no-shift (6.55 vs 5.24), but plain+shift 7.76 > tta; stacking hurts. TTA = weaker alternative to shift. D ttaS3 pending.
- 2026-11-13 15:21: ttaS3 (D) saved+scored: TTA on contrastive ensemble +0.6 pair (noise), AEGIS 36.85 unchanged. TTA closed: stable +1-1.5 pair no-shift on plain ensembles only; shift remains better. All GPUs + CPU idle, queue exhausted.
- 2026-11-13 ~17:20: brainstorm round 11 (ideas.md sec 11): CatchBench, Who&When Pro, AgenTracer; nothing cheap/new to test. Idle: all GPUs, CPU.
- 2026-11-13: wrote local-notes/approach.md (every approach explained with numbers, glossary, pipeline, novelty vs paper sources). Citations from memory; verify before thesis.
- 2026-11-13: loop STOPPED by user request (cron fb04818e deleted). All kernels saved, nothing running. Deliverables: approach.md, results.md, ideas.md.
