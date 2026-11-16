# Field-level gaps, positioning and paper outline (written 2026-11-14 night)

Scope: gaps in the FIELD of failure attribution for LLM multi-agent systems (MAS) that work built on AEGIS can address and that we can realistically implement. Evidence = our own results (local-notes/timeline.md, aegis/results/rr/final_rrf1.json) plus a LIMITED web search done 2026-11-14 (about 6 queries, abstracts and summaries only, not full-paper reading). "To our knowledge" claims below are NOT verified by a systematic search; check before submission. Numbers quoted from other papers come from search summaries and must be checked in the papers.

## Related work found (check each before citing)
- Who&When: Zhang et al., "Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems", ICML 2025, arXiv 2505.00212. Logs from 127 MAS (algorithmically generated + hand-crafted), annotated with the responsible agent and the decisive error step. Methods: All-at-Once, Step-by-Step, Binary Search. Best hand-crafted step-level accuracy reported about 8.77 percent (search summary). Our 184 records = the released records of this benchmark.
- AEGIS: Kong et al., "Aegis: Automated Error Generation and Attribution for Multi-Agent Systems", ICLR 2026, arXiv 2509.14295. 9,533 trajectories made by injecting faults into successful runs, 14 MAST error modes, 6 frameworks; Aegis-SFT / GRPO / contrastive-learning baselines; Who&When used as out-of-distribution test.
- MAST: Cemri et al., "Why Do Multi-Agent LLM Systems Fail?", arXiv 2503.13657. 14 failure modes in 3 groups (specification and system design, inter-agent misalignment, task verification and termination); traces from 7 frameworks, 200+ tasks; LLM annotator validated against experts.
- AgenTracer: Zhang et al., arXiv 2509.03312 (ICLR 2026). Counterfactual replay on failed runs and fault injection on successful runs to build TracerTraj (2,000+ trajectories); Qwen3-8B tracer trained with GRPO; reports prompted LLMs below 10 percent step-level accuracy on hand-crafted Who&When.
- TraceElephant: Chen et al., "Seeing the Whole Elephant", arXiv 2604.22708, ACL 2026. 220 failure traces from 3 systems with full execution traces; full traces improve attribution by up to 76 percent over partial observation; says Who&When was the only dedicated attribution benchmark before it.
- Also seen (not read): hierarchical / causal-graph failure attribution (arXiv 2602.23701), a debugging-LLM-agents study (alphaxiv 2606.00765). Check.

## Gaps (field level) with our evidence and what we did

G1. Injected-fault training signal vs natural failures (transfer gap).
- Field: synthetic faults (AEGIS, AgenTracer) are the scalable supervision; natural failures are scarce and hand-annotated.
- Our evidence: AEGIS test pair F1 about 33 (grouped split) vs Who&When about 8 to 9 (pair), agent about 45 to 48, error about 17 to 23.
- What we did: strict OOD protocol (Who&When never used for training, tuning, thresholds, retrieval index or prompt examples; scored once with a pre-committed config) and two fixes: block-dropout against shortcut features, label-shift correction (transductive, flagged). Who&When pair 3 to 5 before, 8.2 after.
- Not claimed: that this closes the gap.

G2. No leakage-safe evaluation for attribution benchmarks.
- Our evidence: the released train/val/test share task queries; dropping train/val conversations whose query appears in test (4,128 of 8,933 dropped; 3,865 train / 940 val remain) changes a flat MLP from pair 32.19 (paper split) to 23.05 (grouped).
- What we did: query-disjoint grouped split, both splits reported. Verify whether others report this; I found nothing in the search either way.

G3. Cost and accuracy of LLM-based attribution (prompting, judges).
- Field: prompted LLMs are weak (AgenTracer: below 10 percent step-level on hand-crafted; Who&When paper: best 8.77 percent) and fine-tuned 8B tracers need GRPO training.
- Our evidence: a cheap supervised encoder (14 fine-tuned MiniLM + MLP head, no LLM at inference) is competitive; a frozen 7B judge over a top-5 shortlist (about 24 GPU-hours on 8 T4s) added NOTHING when fused (weights 0, both gates failed); judge alone is far below the supervised model on AEGIS (pair about 25 vs 34).
- Exploratory (post hoc, not headline): plain judge alone looked better than the model on Who&When agent F1 (51.7 vs 45.0, interval excludes 0) but not on pair; RAG examples from AEGIS train helped on AEGIS val and did not transfer to Who&When.

G4. Tiny natural benchmark, so noisy comparisons.
- Who&When has 184 records; TraceElephant 220 traces (different label format). One Who&When record is about 0.5 pair points; bootstrap intervals about +-2.5; seed std about +-1.2.
- What we did: report bootstrap CIs, paired differences, halves, and refuse single-seed claims.

G5. No shared label space across attribution datasets.
- AEGIS labels (agent, MAST mode); Who&When labels agent + decisive step (our error-mode scoring uses a mapping, see local-notes/context.md); agent identifiers do not even match textually (about 20 of 184 pairs affected by "Name (role)" vs "Name").
- What we did: separate post-hoc name-normalised row (agent 45.4 to 52.3 for the model alone). Not claimed as a contribution to a standard.

G6. Observation level (partial vs full traces) and long contexts.
- TraceElephant: full traces help a lot. Our judge reads at most about 7k tokens of the conversation (head + tail), a known limitation; our encoders pool per-turn vectors. Step-level attribution is NOT addressed by our pipeline (we predict agent and mode, not the decisive step).

## Honest contribution list (draft)
1. A leakage-safe (query-disjoint) protocol for AEGIS and the measured effect of leakage.
2. A cheap strong baseline: fine-tuned small encoders + MLP head + two-threshold decoder, with ensemble and seed-noise analysis.
3. Block-dropout training and label-shift correction for synthetic-to-natural transfer, with a full ablation (all four masks, with and without shift).
4. A pre-registered shortlist + frozen-judge + retrieved-example study with random-example controls and gates: a documented NEGATIVE result for fusion, plus an exploratory observation on the judge alone.
5. A scoring caveat (agent-name normalisation) reported as a separate, labelled row.
Not contributions: beating published numbers across the board (setups differ: grouped split, transductive shift); step-level attribution.

## Paper outline (IEEE style, bullets)
- Abstract: problem, protocol, three methods/findings, honest negative result, numbers with CIs.
- I. Introduction: MAS failures are structural (MAST); attribution matters for debugging; supervision is synthetic (AEGIS, AgenTracer) while real failures are scarce (Who&When); gaps G1 to G5; contributions 1 to 5.
- II. Related work: Who&When, MAST, AEGIS, AgenTracer, TraceElephant, LLM-as-judge for attribution.
- III. Problem and data: tasks (agent F1, error F1, pair F1, micro/macro), AEGIS-Bench, Who&When, grouped split.
- IV. Method: encoders, per-agent features, MLP head, decoder, block-dropout, label shift; shortlist + judge (negative).
- V. Experiments: ablations (masks x shift), seeds, retrieval block, judge variants (plain/rag/rand/LLM-only), bootstrap CIs, leave-one-framework-out.
- VI. Discussion and limitations: transductive shift, post-hoc rows, 184 records, 4-bit zero-shot 7B, no step-level, dataset switch not yet approved by the supervisor.
- VII. Conclusion.
Verify every citation and number before writing.

## ADDENDUM 2026-11-15: gaps that follow from WHY each method was tried (old HotpotQA/DAG project excluded per user)
H1. Symptom localisation inside an agent's turns: attribution models average over turns, which dilutes the one or two faulty messages. Evidence: gated attention pooling +2.6 to +2.9 AEGIS test pair, Who&When neutral (approach.md 4.8). Gap: how to localise cheaply without step labels. Check whether AEGIS gives the injected turn.
H2. Does interaction structure help transcript-level attribution? Turn-graph GNN vs flat MLP vs 0-layer control. Only the question node helped (frozen), gone after fine-tuning, floor on Who&When. Gap: structure-aware claims lack a no-structure control; free-form transcripts do not expose wiring. Contribution is a controlled NEGATIVE result.
H3. Order-aware (self-attention) encoders overfit framework style: --seq collapsed AEGIS test pair to 19-22 while Who&When agent F1 rose to 50-53. Gap: order/sequence models vs framework shortcuts; frame together with G1.
H4. Shortcut cues (agent names, framework style, peer statistics) that do not transfer: block-dropout and gradient-reversal training. Gap: attribution evaluation does not test for name/framework shortcuts.
H5. Does a zero-shot LLM judge add to a supervised attributor? Pre-registered, 2 studies (k=5, k=12): fusion weight 0, no gain beyond noise; RAG helped AEGIS val only. Gap open for larger judges (not tested). Contribution is a negative result with controls.
H6. Label-prior shift between synthetic and natural failures: label-shift correction gives +2.7 pair on Who&When (transductive).
Caveat: "gap" here means what our evidence points to; absence in the literature is NOT verified.

## LITERATURE CHECK 2026-11-15 (7 web searches + 2 abstract fetches; snippets/abstracts only, NOT full papers; verify before citing)
- H2 GNN: NOT open. AFANet "Beyond LLM-Based Reasoning: Lightweight GNNs for Agent Failure Attribution" (arXiv 2608.18575): turn-level conversation graph, claims to match or beat LLM baselines incl. fine-tuned, robust across GNN architectures, cheap test-time adaptation on an OOD benchmark; abstract names no datasets, numbers or flat-baseline ablation (check full text). Also CHIEF (2602.23701, causal graphs), GraphTracer (2510.10581, information dependency graphs), TraJudger (structure-guided). Our angle left: controlled flat vs 0-layer-control vs GNN on AEGIS shows no gain after fine-tuning (counter-evidence), if their ablation lacks that control.
- H6 label shift / test-time adaptation: AFANet reports test-time adaptation on OOD, so adaptation itself is not new; our specific label-prior (BBSE-style) correction for synthetic -> natural not found in search.
- H5 judge: judges are well covered (AgentLocate 2607.07989: LLM judge + independent evaluators + lightweight fine-tuning; DuoTrace "Detect Before You Attribute" 2608.29646; fine-tuned judges incl. AEGIS, AgenTracer). NOT found: fusing a frozen judge with a supervised attributor's probabilities under pre-registered gates with random-example control. Narrow claim only.
- H1 attention / localisation: related, not identical: SOAP (soft dependency graph from attention, no step labels), one-class step anomaly scoring (2607.12747), AgentLocate (agent + earliest step). MIL attention pooling per agent not found, but it is already in DCL (AEGIS paper) -> not novel; only an ablation.
- H3 sequence / H4 shortcuts (names, framework style): no paper found testing it; search was thin. Who&When Pro (injected errors, 15 frameworks, 26 benchmarks, 12,326 traces) is the closest benchmark; check whether it analyses cross-framework / shortcut effects.
- G2 leakage / grouped split: no attribution paper found with a query-disjoint split; other fields do grouped splits (fall detection, pathology, ship acoustics). Plausibly open, unverified.
- G1 synthetic vs natural transfer: AEGIS, AgenTracer, Who&When Pro all use injection; the transfer gap itself is acknowledged (AEGIS uses Who&When as OOD). Our contribution is a protocol + analysis, not the observation.

## AFANet full-text read (arxiv.org/html/2608.18575, via WebFetch summary, first 100k chars; numbers need checking in the PDF)
- Data: AEGIS-Bench released split 7,146/1,787/600 (leaky protocol, no query-grouping mentioned) + Who&When 184 as OOD, zero training on it.
- Graph: node = turn; features = TF-IDF/SVD deviation features + stats + frozen all-MiniLM-L6-v2; edges = temporal + same-agent (no question node); 2-layer GCN (GAT/SAGE also tried).
- Ablation "W/o GNN": avg 22.24 vs 24.82 full; Who&When pair micro 2.87 vs 6.90. Edge-type ablations small (22.1 to 23.6). No MLP baseline, no fine-tuned encoder.
- Who&When AFANet main: agent 37.93, error 13.79, pair 6.90 (TTA table base 5.17, unexplained); TTA = TENT-style entropy minimisation on LayerNorm, pair-level TTA pair 7.47.
- Differences from ours: frozen encoder only; no question node; no 0-layer control with matched params; no fine-tuning; paper split (leaky); entropy-min TTA vs our label-prior shift. Our fine-tuned graph = no gain over flat. Not the same work; partly overlapping and partly contradicting (they report graph helps, frozen).

## WHOLE-FIELD (multi-agent LLM systems) GAPS, 2026-11-15 (3 web searches, snippets only; many sources are preprints/blogs; verify before citing)
F1 Failures are mostly structural/coordination (MAST 2503.13657) but diagnosis is left to LLM judges; no cheap system-level diagnosis. Ours: AEGIS mode+agent attribution, CPU encoder.
F2 Monitoring/failure detection is per agent, system-level breakdowns missed (LumiMAS). Ours: joint scoring of all agents in a trajectory with peer features (helps in-domain only).
F3 Debugging = resample vs repair: Repair or Resample (alphaxiv 2608.25920): reruns reproduce failures about 68%, repair about 7%. Gap: does correct attribution make repair better? NOT testable with our data (needs rerunning MAS) -> future work.
F4 Evaluation inconsistent/narrow ("evaluation vacuum" in survey fi18060326; incomparable configs). Ours: leakage-safe split, CIs, seeds, harmonised scoring.
F5 Benchmarks miss silent/production failures (2606.14589: unit of analysis = incident, not task trace). Ours: injected (AEGIS) -> natural (Who&When) transfer gap measured: pair about 33 vs about 8.
F6 Observability not coupled with controlled fault injection (2608.24271). Ours: AEGIS is injection; we test whether detectors trained on injection transfer.
F7 Fault tolerance needs low latency/cost (2511.20663 family, COCO). Ours: cost table (encoder minutes CPU vs judge about 24 GPU-h with no gain).
F8 Topology and error amplification (Google scaling study, secondary blog, unverified). Ours: turn-graph negative result only.

## THESIS FOCUS (user decision 2026-11-15): attention methods + GNN only; gaps told as motivation
Title idea: "Does conversation structure help failure attribution in LLM multi-agent systems? A controlled study of attention pooling, sequence attention and turn graphs on AEGIS-Bench."
Gap chain (field -> specific -> ours):
 1. Field: MAS failures are structural/coordination (MAST), diagnosis needs cheap system-level methods (F1, F2).
 2. Specific: structure-aware attributors (graphs, attention) are proposed, but (a) no controlled comparison of turn aggregation (mean vs gated attention vs self-attention vs graph), (b) AFANet's only ablation removes message passing; no flat / parameter-matched control, frozen encoder only, released leaky split, (c) transfer to natural failures not tested leakage-safe.
 3. Ours: controlled study, grouped + paper split, frozen vs fine-tuned encoder, seeds + CIs, Who&When OOD once.
Findings to report honestly: gated attention +2.6 to +2.9 AEGIS pair, Who&When neutral; self-attention (--seq) collapses AEGIS to 19-22; turn-graph gain only from question node, gone after fine-tuning (within noise), at floor on Who&When; layout of numbers in results_master.md section 10 and approach.md 4.8/4.10.
Risk: mostly negative/within-noise; novelty = controlled evidence, not a new model. Block-dropout / label shift (the only Who&When gains) are outside this focus: either include as 'what actually helped transfer' or state they are outside scope.
