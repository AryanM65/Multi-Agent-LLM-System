# Paper review: AEGIS (arXiv 2509.14295v6, ICLR 2026)

Read on 2026-11-12: main text and all appendices. Table 1 was re-extracted with `pdftotext -raw` because the layout extract garbled it.
Local only. Never commit or push this file.

## 1. Errors and inconsistencies in the paper
1. Section 7 says Claude-Sonnet-4 (18.16) is third among general-purpose LLMs. Table 1 gives Gemini-2.5-Pro 18.35, so Pro is third.
2. Table 2: the six task counts (2048, 1871, 1741, 1497, 1446, 954) sum to 9,557, not 9,533. The framework counts (sum 9,533) and error-mode counts (sum 24,843, about 2.6 per trajectory) are consistent.
3. Table 3 lists 5 of the 6 MAS frameworks (SmolAgents missing).
4. SFT epochs: Section 7 selects 2 epochs, Appendix D.1 says 3.
5. DCL hardware: Section 6 says 4090 GPUs, Appendix D.1 says 3090 GPUs.
6. Table 1 Avg column: Qwen2.5-14B-Instruct reported 13.99 (computes to 14.02), o3 reported 20.24 (computes to 20.28), Claude-Sonnet-4 18.16 (18.17). The other rows match.
7. Appendix D.1 calls GRPO "reinforcement learning with human feedback" and SFT "mathematical error detection"; neither fits the method.
8. Appendix D.2 cites Pair r = -0.106, which is Pair macro; Pair micro is 0.291 (Table 11).
9. Generalisation to Who&When is thin. Qwen2.5-14B base to Aegis-SFT: Who&When agent micro 49.88 to 51.14; Who&When error micro Aegis-SFT 9.87 versus o3 14.88; pair micro 4.03 versus 8.10 (Qwen3-8B-Thinking + GRPO).

## 2. Missing from the paper
- No trivial floor. "Random" averages 4.08, while "predict every present agent with all 14 modes" averages about 25.6 on our test floor. Aegis-SFT (26.51) is barely above it. DCL and the zero-shot rows sit below the floor on agent and error micro.
- No mention that a random split leaks (several faulty variants share one query). We measured 588 of 600 test queries also in train.
- Random baseline sampler not described; no seeds for LLM rows (only DCL has 3 seeds); no accuracy check of the Gemini-2.5-Flash mapping used for Who&When labels; no DCL code or checkpoint; no evaluated-set size for Who&When (v6 gives 184 annotated errors for the original benchmark).

## 3. What it means for us
- Fine-tuning MiniLM end to end is the paper's own DCL recipe (lr 1e-4, 2 epochs x 500 steps, batch 128, top-K 3, 384-d encoder, loss weights 1.0/0.4/0.6, ASL, tau 0.07). Our fine-tune gain replicates the paper's practice. It is not a new contribution.
- Our DCL-style model is not DCL: frozen encoder, no clean-turn positives from the original successful trajectory.
- Fair comparison is pair level: DCL 8.33 (published) versus our fine-tuned 28.5 (grouped test) and 33-36 (paper split, leaky like the paper's own). Agent and error micro gaps are floor-driven.
- DCL published numbers (Table 10, 3 seeds): Aegis-Bench pair 8.33 +- 1.84 / 5.30, agent 22.93 / 20.23, error 24.73 / 27.70; Who&When 1.60 / 0.77, 8.40 / 6.07, 14.67 / 10.57; average 12.61.

## 4. Fixes applied (2026-11-12)
The earlier "paper says 127 Who&When trajectories" note is not supported by v6 (it says 184 annotated errors and gives no evaluated-set size). Corrected in: `aegis/docs/dataset_audit.md`, `aegis/docs/decisions.md` (D5), `aegis/docs/reproduction_log.md`, `local-notes/file.md`, `local-notes/context.md`, `new-plan.md`, root `context.md`.

## 5. Not yet verified
- Which 600-record test file the paper used (HF file versus `evaluation/data/aegis.jsonl`).
- Whether released records contain the clean successful trajectory needed for DCL-style contrastive positives.
