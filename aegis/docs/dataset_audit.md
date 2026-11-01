# Aegis-Bench dataset audit (Phase 1)

Date: 2026-11-01. Sources: HF `Fancylalala/AEGIS` (232 MB, in `aegis/data/raw/`) and GitHub
`kfq20/AEGIS` (shallow clone in `aegis/third_party/AEGIS/`). Both are gitignored.

## Answers to PLAN.md section 4

1. **Repo contents.** Present: `evaluation/evaluate.py` (authors' metric script), `evaluation/prompt.txt`
   (standard prompt), `prompt_cot.txt`, `api_model.py`, `local_model.py` (inference), `evaluation/data/{aegis,whowhen}.jsonl`,
   `verl/` (SFT/GRPO training), `data_processing/dataset_builders/`. **Absent: any DCL code or checkpoint**
   (grep for prototype / contrastive / DCL / multiple-instance finds only `build_contrastive_pairs_dataset.py`,
   a dataset builder). Consequence: Phase 3b falls to the "DCL-style re-implementation" branch.
2. **Schema.** Each record: `id`, `metadata{framework, benchmark, model, num_agents, num_injected_agents, task_type}`,
   `input{query, conversation_history[], final_output?}`, `output{faulty_agents[{agent_name, error_type, injection_strategy}]}`,
   `ground_truth{correct_answer, injected_agents[], is_injection_successful}`.
   The label lives in `output.faulty_agents` (not `ground_truth`).
3. **Topology edges: not stored.** No edge or adjacency field anywhere. Graph option **B (turn graph)** or an
   agent graph built from conversation order; framework topology must be inferred. See decisions.md D3.
4. **Conversation format.** `conversation_history` is a list of turns `{step, agent_name, content, phase, agent_role?, is_injected?}`.
   129 train turns have `content = None` (treated as empty string). Mean 11.9 turns per trajectory (median 11, max 45).
   Characters per trajectory: mean 19.8k, median 17.2k, p95 36.5k, max 285k.
5. **Who&When.** A processed file IS distributed: `evaluation/data/whowhen.jsonl`, 184 records, already mapped to the
   14 modes, 1 label pair per record. The paper says 127 trajectories; the file has 184 (unexplained; see decisions D5).
   In 20 of 184 labels the faulty agent name does not appear in the turns.
6. **Leakage in `metadata`/turns (important).**
   - `metadata.num_injected_agents` reveals the number of faulty agents: forbidden.
   - `ground_truth.*` (injected_agents, is_injection_successful): forbidden.
   - Turn field `is_injected` exists **only for macnet** (17,496 of 84,758 train turns, 100% of macnet turns, 0% elsewhere).
     Its agent set equals the label agent set in only 839 / 7146 trajectories, so it is not a perfect label copy, but it is
     still injection metadata: forbidden. The loader never reads it.
   - `agent_role`, `phase`, `step` are not injection information; `phase` and `agent_role` are allowed features.
   - Not yet checked: text in `content` that literally reveals the injected instruction (to do in Phase 4 error analysis).
7. **Train/test base-task overlap: severe leakage.** 588 of 600 test records have a query that also appears in train
   (430 of 438 unique test queries). 1753 of 1787 val records share a query with train. Mean records per query is 5.6 (max 56).
   Released train/val/test splits are therefore NOT grouped by task. See decisions D2.

## Counts

| Split | Records |
|---|---|
| train | 7146 |
| val | 1787 |
| test (Aegis-Bench) | 600 |
| who_when | 184 |
| grouped train / val (test queries removed, split by query) | 3865 / 940 |

Total 9533 matches the paper. Test frameworks: macnet 156, agentverse 143, llm_debate 140, dylan 85, smoagents 48, magentic_one 28.
Test benchmarks: GSM8K 105, HumanEval 105, MATH 105, MMLU 105, SciBench 104, GAIA 76 (so the test set is not exactly 100 per task).
Train frameworks: agentverse 1821, llm_debate 1816, macnet 1749, dylan 1198, smoagents 360, magentic_one 202.

## Labels

- Every record has at least one label. Train mean 2.38 pairs and 2.32 distinct agents per trajectory. 12 duplicate pairs in train.
- All label agent names appear among the trajectory's turn agent names (0 of 17,007 missing in train).
- Train mode counts: FM-1.1 1632, 1.2 1326, 1.3 1063, 1.4 1249, 1.5 1523, 2.1 1192, 2.2 1275, 2.3 1326, 2.4 1085, 2.5 1077,
  2.6 1012, 3.1 1004, 3.2 1186, 3.3 1057 (roughly balanced).
- Only `FM-x.y` codes occur; `normalize_mode` also accepts `EM-x.y`.

## Agent-name formats (the agent-level metric uses these raw strings as classes)

- agentverse: Solver, Critic, Evaluator, RoleAssigner (4)
- llm_debate: User, Assistant1-3 (4)
- dylan: Assistant1-4 plus role names (AlgorithmDeveloper, Programmer, ComputerScientist, PythonAssistant, Tester, Reflector, ...) (15 distinct)
- macnet: Node-2, Node-1, Node0, Node1, Node2, Node3, Node4 (7; the name `Node-1` is a real label, not a negative number)
- magentic_one: MagenticOneOrchestrator, WebSurfer, user, Coder, FileSurfer, ComputerTerminal (6)
- smoagents: Searcher, Manager, User, System (4)
- 31 distinct label names in total. Names like `Assistant1` are shared by dylan and llm_debate, so agent-level classes
  mix frameworks. Who&When uses free-form names (`Orchestrator (thought)`, `WebSurfer`, `Verification_Expert`, ...).

## Evaluation data discrepancy

`evaluation/data/aegis.jsonl` (600 records) has **zero id overlap** with the HF `Aegis-Bench/test.jsonl` (also 600),
382 shared queries, and 585 of 600 queries also occur in train. The paper's test set is one of these two; we cannot tell
which. We use the HF test file as the primary test set (D1).

## Not yet done

- Reading 5 raw trajectories per framework next to their labels (do in Phase 4 together with error analysis).
- Searching content for literal injection strings.
- Token-length distribution in tokens (only characters measured so far); needed before the Phase 3a zero-shot run.
