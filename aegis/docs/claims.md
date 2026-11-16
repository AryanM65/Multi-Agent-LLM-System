# Claims register (S5, written 2026-11-16)

Every claim the paper may make, with its evidence and its novelty status. "Not found" means NOT FOUND IN OUR SEARCH, which was limited
(WebFetch summaries of papers and web-search snippets, not systematic reading). Papers must say "to our knowledge, we did not find",
never "no prior work". Re-check before submission.

## What was read in S5 (limits)
- AFANet, arXiv 2608.18575 (HTML): first 100,000 of 102,999 characters via a summarising fetch. Answers below are from that text.
- Who&When Pro, arXiv 2607.09996 (HTML): first 100,000 of 200,975 characters; appendices NOT seen. "Not found" there only covers the main text.
- Two extra web searches (label-shift/EM prior for attribution; duplicate/leakage/shortcut audits of AEGIS-Bench).

## Status of each claim

| # | Claim | Evidence | Novelty check | Status |
|---|---|---|---|---|
| C1 | The released AEGIS split leaks: 588/600 (HF test) test records share a query with train, and 123 (20.5%) HF test conversations are verbatim copies of train records; removing duplicates lowers pair micro-F1 by 6.1 (repo file) to 9.2 (HF file) | Our own data: `aegis/results/test_file_comparison.json`, `s2_*` and `block_ablation.json` | No leakage or duplicate audit of AEGIS found (search 2026-11-16). The AEGIS paper says its subsets are "distinct, non-overlapping"; our finding is about the RELEASED files and conversation text, so phrase it that way | KEEP, headline |
| C2 | A model that never reads the conversation (agent name + framework + turn counts) reaches 10.75 [9.2, 12.4] pair micro-F1 on the authors' file (dedup) and 14.30 on the HF file (released), above published DCL (8.33) and every published zero-shot LLM (max 7.86) | `block_ablation.json`, `comparison_tables.md`; frozen-encoder MLP only | No shortcut-only baseline on AEGIS found. CatchBench (2608.22808) and an audit (cited as [12] in the draft) do shortcut checks on OTHER benchmarks | KEEP, headline; wording: comparison is to numbers on a test file the paper does not name |
| C3 | The authors' repo file `aegis.jsonl` is the test set the paper describes (100 per benchmark); HF `test.jsonl` is not (105/105/105/105/104, GAIA 76) | Our counts; paper Section 6 text. A later paper repeats the description (search snippet) | n/a (our observation) | KEEP; say "matches the description", not "is the file used" |
| C4 | Label-shift correction (EM prior estimation, Saerens et al.) narrows the injected-to-natural gap on Who&When **only when the name block is hidden** (names-only 6.16 -> 10.68 [8.6, 12.7]; all blocks on 4.96 -> 5.14, no effect; peer-only hurts) | S4 ensemble under the published protocol, `aegis/results/s4_ensemble.json` (re-run 2026-11-16); contradicts the draft's all-blocks claim (5.56 -> 8.24) | AFANet: no label shift, uses TENT-style entropy minimisation on LayerNorm params with unlabelled OOD data (so "test-time adaptation" is not new, label-prior correction not found there). Who&When Pro main text: no shift analysis. Search: prior-shift correction exists for agent OUTCOME prediction (arXiv 2609.25647, oracle priors), not for attribution | KEEP with wording "we did not find EM label-shift correction for failure attribution"; cite AFANet TTA and 2609.25647 as related |
| C5 | Block-dropout of name/peer/framework feature blocks improves transfer | S4: single model serves all four masks; Who&When agent micro-F1 49.9-51.3 vs 45.2-46.2 for peer-only/neither; pair gain mostly within intervals | AFANet ablates feature blocks at TRAINING time only, no masking/dropout, no name-shortcut analysis. Not found elsewhere | KEEP, mark as needing S4 re-run |
| C6 | A turn-graph model gives no gain over a parameter-matched flat/0-layer control after fine-tuning; only the question node helped when frozen | `aegis/results/metrics_history.jsonl`, `local-notes/results_master.md` section 10 | AFANet reports its graph helps (frozen encoder, no flat or matched-parameter control, no query node); its Who&When numbers differ between its Table 1 (pair 6.90) and Table 5 base (5.17) | KEEP as controlled negative result; compare to AFANet only as reported |
| C7 | A frozen 7B judge adds nothing on top of the trained model (two pre-registered studies) | `aegis/results/rr/`, `rr_k12/` | Judges are well covered elsewhere; fusion under pre-registered gates with random-example control not found | KEEP, narrow wording |
| C8 | Published agent-level numbers sit below the trivial floor | Table 1 vs our floor | n/a | OBSERVATION ONLY, no claim built on it (guide's rule: do not explain published results) |
| C10 | 14-encoder ensemble, published protocol, authors' file: pair 35.77 [33.1, 38.2], agent 73.20, error 45.79 vs published Aegis-SFT 16.62 / 76.53 / 27.53 (agent is lower than Aegis-SFT) | `s4_ensemble.json` | n/a | KEEP; mention test duplicates were removed from OUR training data only (published models trained with them), a conservative choice |
| C9 | Who&When: our simple full model is at or below the trivial floor (pair 1.85 [0.8, 3.2] vs floor 3.42); shortcut-only 4.75 [2.6, 7.0] | `block_ablation.json` | n/a | KEEP; the draft's 8.24 (ensemble + block-dropout + label shift) is NOT reproduced yet |

## Consequences for the draft
1. Lead with C1-C3 (verified here, intervals, committed). They need no GPU work to defend.
2. C4, C5, C7 numbers in the draft rely on runs not reproduced on this machine: they need S4 or must be labelled "as logged on Kaggle".
3. Remove "Our protocol differs from [7]" hedging only after S4 reproduces the Who&When rows.
