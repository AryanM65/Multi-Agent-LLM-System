# Decisions log (Aegis track)

## 2026-11-01 D1: Primary test set = HF `Aegis-Bench/test.jsonl`
Reason: it is the official release (600). The repo's `evaluation/data/aegis.jsonl` is a different 600 (0 id overlap).
Which one the paper used is unknown. If time allows, also report on the second file.

## 2026-11-01 D2: Report two splits, "paper" and "grouped"
Reason: 588/600 test queries also occur in train (audit item 7). "paper" = released split (comparable to published numbers,
likely inflated). "grouped" = test unchanged, train/val with every test query removed and re-split by query (seed 0, 80/20):
3865 train / 940 val. Cost: 46% less training data. Both reported; claims about generalization rest on "grouped".

## 2026-11-01 D3: Graph construction
No edges are stored. Provisional: option B (turn graph; edges between consecutive turns, both directions, and between
turns of the same agent), pooled per agent. Revisit in Phase 5.

## 2026-11-01 D4: Metric definition
Ported from the authors' `evaluate.py` (`aegis/src/metrics.py`). Macro-F1 averages per-class F1 over the union of
classes seen in true or predicted sets; agent-level classes are raw agent-name strings. Agent names are only whitespace-stripped,
not case-folded, to stay identical to the authors' scoring. Unparseable predictions count as empty (the authors skip them).
Unit tests in `aegis/tests/test_metrics.py`.

## 2026-11-01 D5: Who&When = `evaluation/data/whowhen.jsonl` (184 records)
Already mapped to the 14 modes, so no local LLM mapping is needed. The paper says 127; unexplained, report the count honestly.
20/184 label agents do not occur in the turns, so those can never be predicted from the graph; report this ceiling.

## 2026-11-01 D6: Leakage whitelist
`parse_record` reads only: id, metadata.framework, metadata.benchmark, input.query, turn step/agent_name/content, output labels.
Forbidden: `is_injected`, `ground_truth.*`, `metadata.num_injected_agents`, `metadata.num_agents`, `injection_strategy`.
`phase` and `agent_role` are allowed later as features but are not loaded yet.

## 2026-11-01 D7: Random baseline definition
Not in the released code. Agents drawn from the global pool of 31 names, 1 pair (70%) or 2 pairs (30%), fitted by hand to
the paper. Result (10 seeds, test): pair 0.40/0.19, agent 5.86/4.03, error 11.74/11.53 versus paper 0.33/0.21, 4.54/3.56, 11.23/11.15.
Verdict: **close, not matched** (agent micro is 1.3 points high; the paper's exact sampler is unknown). Phase 2 is accepted on
the strength of the unit tests plus this approximate match. The knobs were fitted to the paper, so it is a harness check, not a baseline.

## 2026-11-01 D8: No DCL code released
Phase 3b will be a re-implementation from Appendix C / Table 7 and must be named "DCL-style".

## 2026-11-02 D9: Embeddings computed on Kaggle GPU, cached locally
A local full-corpus MiniLM run was killed by the OS for low memory (nothing lost, no shard was written). Embeddings are now produced by
`scripts/kaggle_embed_minilm.py` (99,406 unique turn texts, 275 s on GPU), shards in `aegis/data/cache/` (gitignored), keyed by sha1 of the
turn text. Every turn of train/val/test/who_when is covered (verified: 0 missing). Frozen all-MiniLM-L6-v2, max_seq_length 256.

## 2026-11-02 D10: Decision rule and report conventions for Phase 4+
Pair predicted iff p_agent >= t_a and p_mode >= t_m, at least one pair per trajectory (every record has a fault), thresholds tuned on val only.
Every results table includes the trivial "predict everything" floor and the frequency baseline, because agent-level micro-F1 has a floor near 70
(54-56% of agents are faulty) and error-level micro-F1 a floor near 28. Published DCL (22.9 / 24.7) is below both floors.
