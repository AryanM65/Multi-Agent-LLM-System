# Statements in `Failure_Attribution_Transfer_IEEE_v2_12pages (1).pdf` to fix before the paper is rewritten (2026-11-16)

Source for each fix: the evidence file named in brackets. Section numbers refer to the draft. Nothing here has been applied to the PDF (no LaTeX source is in the repo);
this list drives the rewrite.

## A. Comparisons to published rows that are not like-for-like (the main rejection risk)
1. **Abstract and V-A, Table V.** "32.96 pair against 16.62 for the best published trained model on the same 600-record test set": ours is trained on the query-disjoint
   split (3,865 train) and scored on the HF test file; the published models were trained on the released split and the paper does not name its test file.
   FIX: headline row = published protocol (released train/val, test duplicates removed) scored on the authors' repo file `aegis.jsonl` [S4 ensemble, `aegis/results/s4_ensemble.json`];
   the query-disjoint number stays as a separate, labelled row; give both test files. [decision D11, `aegis/results/comparison_tables.md`]
2. **Table V caption and V-A text** say "the same 600-record AEGIS-Bench test set". FIX: name the file; state that the authors' file matches the paper's description (100 per benchmark)
   and the HF file does not (decision D11).
3. **V-A, scorer caution** ("we re-implemented the scorer ... should be confirmed by running both scorers on identical outputs"). FIX: replace with the parity result: same F1 code
   on 200 fuzz sets and on end-to-end JSONL, to 1e-12 [`aegis/tests/test_scorer_parity.py`]; only difference is the unparseable-output rule (authors skip, we score empty).
4. **V-F(i)** "The floor and the published rows come from different scorers, so these are indications." FIX: the scorers agree (see 3); the floor is our computation and is
   file-dependent (7.35 HF file, 6.47 authors' file). Report both floors.
5. **Abstract and V-F(i)**: "published zero-shot LLMs score at most 7.86 pair micro-F1, close to a trivial floor of 7.35": 7.35 is the HF-file floor; on the authors' file it is 6.47. Give both.

## B. Leakage statements that are now understated
6. **Contribution 1 and III-D** say query overlap inflates "by more than 12 pair points". Keep, and ADD the stronger finding: 123 of the 600 HF test conversations (20.5%) are verbatim
   copies of train records (122 with identical labels) and 108 of validation; removing duplicates cuts pair micro-F1 by 6.1 (authors' file) to 9.2 (HF file) for the same model
   [`aegis/results/test_file_comparison.json`, `block_ablation.json`]. The AEGIS paper says its subsets are non-overlapping: phrase as "in the released files".
7. **III-D**: add that the HF train/val also overlaps the authors' test file (585 of 600 records share a query with HF train; 125 verbatim conversations).

## C. New results to add (they are verified and need no GPU to defend)
8. **Shortcut-only baseline** (agent name, framework, turn counts; never reads the text): pair micro-F1 10.75 [9.2, 12.4] on the authors' file after duplicate removal, 14.30 on the HF file
   under the released protocol; above published DCL (8.33) and every published zero-shot LLM (max 7.86). Content-only 16.79. [`block_ablation.json`, `comparison_tables.md`]
   Caveat to state: frozen-encoder model; published rows are on a test file the paper does not name.
9. **Who&When**: our frozen-encoder full model is below the floor (1.85 [0.8, 3.2] vs 3.42); shortcut-only scores 4.75 [2.6, 7.0]. Say so; the ensemble with block-dropout and label shift is the
   method that moves it [S4 ensemble].

## D. Numbers that must be replaced by S4 results (or labelled "query-disjoint, as logged")
10. Tables V, VI, VII and Figs 2-3 (32.96, 5.56, 8.24 and the mask grid) were computed on the query-disjoint split with the HF test file. Replace the headline versions with the S4 ensemble
    (published protocol, authors' file primary) and keep the old ones as a labelled query-disjoint column. [`aegis/results/s4_ensemble.json`]
11. **Table IV**: "631 of 3,865 training records held out" is the query-disjoint run; S4 holds out 1,068 of 6,912 (15%). Per-epoch time about 225 s on a T4 (draft says about 150 s).
12. **Abstract**: "Block-dropout ... and label-shift correction ... help": re-confirm on the S4 ensemble before keeping the wording; if a row no longer shows the effect, say so.

## D2. S4 result that CONTRADICTS the draft (added 2026-11-16 after the 14-encoder run, `aegis/results/s4_ensemble.json`)
16. **Label shift with all blocks on does not help under the published protocol**: Who&When pair micro-F1 4.96 [2.6, 7.6] -> 5.14 [3.2, 7.1] (the draft: 5.56 -> 8.24). Shift helps clearly only when the PEER block is hidden
    (name block on, peer off: 6.16 -> 10.68 [8.6, 12.7]; both off: 4.77 -> 6.67) and hurts when names are hidden but peer is on (4.71 -> 3.78). (Mask key `names` = name block ON, peer OFF.) The draft says shift "raised Who&When pair micro-F1 on every ensemble that kept the name block" and "hurt when only peer features
    (or neither block) were available": both statements must be rewritten from the S4 grid. The name-on/peer-off mask was selected after seeing Who&When (post hoc); leave-one-framework-out favoured all blocks on; so that mask + shift is an
    exploratory lead, not the headline. Abstract, contributions, RQ3 outcome (Table XV), Discussion ("Together with ensembling, they take Who&When pair F1 ... to about 8") and Conclusion all need this change.
17. **Agent-level on the authors' file** (73.20) is below published Aegis-SFT (76.53) although above the floor (62.00); state it next to the pair/error gains.

## E. Citations and claims that must be re-verified (not checked here)
13. Related work claims about audits [11]-[13] (e.g. "simple predictors match or beat GPT-4.1 on three benchmarks", "a NeurIPS 2026 study of 24,000 traces"): taken from search snippets earlier;
    read the sources before keeping any sentence that quotes numbers. V-F(ii) leans on [12].
14. Related-work wording "we did not find": keep that phrasing (claims register `aegis/docs/claims.md`); never "no prior work".
15. AFANet numbers in Table I/text: Table 1 pair 6.90 vs Table 5 base 5.17 are inconsistent inside AFANet; cite as reported.

## F. Things to leave alone
- The frozen-judge negative result (Table XI-XIII), the framework-adversarial negative result (Table X), the leave-one-framework-out result (Table IX): not touched by these fixes
  (their logs are in `aegis/results/rr*/` and the ledger). They stay as logged, and are labelled "query-disjoint".
- Published rows copied from Table 1 of [7] are verified against the paper [`aegis/results/published_rows.json`].
