# Follow-up run "k12" (plain judge, top-12 shortlist): pre-commit note, written BEFORE the judge runs (2026-11-15 ~07:15)

Origin: in the first judge study (final_rrf1) the plain judge ALONE looked better than the model on Who&When agent F1 (51.66 vs 45.00, paired +6.65 [+3.54, +9.88]) while pair F1 was +1.82 [-0.19, +3.74]. That was read from a table of 8 rows on the same 184 records, so it is a post-hoc, exploratory observation. The headline of that study (shortlist_model, fixed rule) showed no gain.

Question of this run: does a plain frozen 7B judge, given a LONGER shortlist (top-12, Who&When recall@12 = 52.7 percent vs recall@5 = 28.8 percent), beat the model alone on Who&When, when its weights and threshold are fixed on AEGIS validation only?

Fixed before any Who&When score is computed:
- Judge: Qwen2.5-7B-Instruct, 4-bit, variant plain only, option-order swap, same prompt and prefix budget as before (char budget 20000, max 7000 prefix tokens, prefill chunk 1024).
- Shortlist: top-12 (agent, mode) pairs per record by p_agent*p_mode from dump_rr_base.npz (candidate tag rrk12; examples not used). Judged splits: AEGIS val (940) and Who&When (184). AEGIS test is NOT judged in this run.
- Tuning: rr_fuse --phase tune on AEGIS val only (2-fold cross-fitting by query hash, one threshold on all val), tag rrf2, variants plain. Headline row by the same fixed rule as before (best cross-fit validation objective among shortlist_model, plain).
- Gates as before (Gate 1: fused plain beats shortlist_model on held-out frameworks). Gate 2 (rag vs rand) does not apply (plain only).
- Final: rr_fuse --phase final --final-splits who_when, ONCE (refuses to overwrite).
- Success bar (stated now): the headline row's Who&When pair F1 is above the model-alone row with a paired bootstrap interval that excludes 0, AND the AEGIS-val validation objective of the headline row is not below shortlist_model. Agent-F1-only gains are reported but do not meet the bar.
- Caveats that stay: same 184 records as the first study (this is NOT an independent test set; a gain would still need an external replication), 4-bit zero-shot 7B, label shift transductive for the model probabilities, the first-study observation motivated this run.
- Outputs go to aegis/results/rr_k12/ (RR_DIR redirect), so the judge files cannot collide with the k=5 ones; because RR_DIR is set the ledger is not written (rows are in final_rrf2.json and the log).
