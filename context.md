# Project context (read this first)

This file explains what the project is, why it changed direction, and the vocabulary used in the other documents.
The day-by-day record of what was actually done is in `current-progress.md`.

## 1. What the project is

This is the BTech thesis project (BTP-I, Group 121, NSUT) of Aryan Mahajan.

A *multi-agent LLM system* is a group of language-model "agents" that talk to each other to solve a task (for example a math
problem, a coding problem or a web-research question). Sometimes the group gets the answer wrong because one or more agents
made a mistake. The thesis task is **failure attribution**:

> Given the full conversation of a multi-agent run, say **which agents were faulty** and **which kind of mistake each one made**.

The kinds of mistakes come from a fixed list of 14 error modes, named FM-1.1 to FM-3.3 (for example FM-1.1 is an agent
not following the task specification; the full list is in `aegis/src/schema.py`). A prediction is a set of pairs
`(agent name, error mode)`.

## 2. Why the direction changed

The first version of the project (folders `btp-pipeline/` and `model/`, branches such as `model_v1`) generated its own
training data by injecting faults into agent systems and trained a graph neural network on it. `new-plan.md` switched the
project to a **published benchmark** so the results can be compared with other people's numbers instead of only with our
own generated data.

- The old pipeline and its results were **not touched** by the new work. They are still in `btp-pipeline/`, `model/`, `dataset/` and `docs/`.
- All new work lives in `aegis/` on the branch `aegis_v1` (and is copied to the branch `current-progress`).
- The switch itself has to be approved by the project guide. That approval has **not** been recorded yet, so everything from
  Phase 5 onward (the graph model) is labelled exploratory in the logs.

## 3. The research paper used

**AEGIS-Bench**, from the paper by Kong et al., ICLR 2026, arXiv **2509.14295** (the AEGIS paper and its released code and data).

What we take from it:
- the dataset (AEGIS-Bench: 9,533 conversations from six multi-agent frameworks, with labelled faulty agents and error modes),
- the evaluation code (`evaluation/evaluate.py`), which we ported exactly so our F1 numbers mean the same thing,
- the zero-shot prompt (`evaluation/prompt.txt`) and the processed Who&When file (`evaluation/data/whowhen.jsonl`),
- the published result rows (Table 1 of the paper) that we try to reproduce or compare against, including the paper's small
  baseline called **DCL** (a MiniLM-based model with a multiple-instance attention head). The authors did **not** release DCL code.

The authors' repository is cloned into `aegis/third_party/AEGIS/` (git-ignored, so it is not pushed). The Hugging Face dataset
files go in `aegis/data/` (also git-ignored).

Other tools and models named in the work (these are third-party and are used as-is):
- **Qwen2.5-7B-Instruct (AWQ 4-bit)** run with vLLM, for the zero-shot reproduction.
- **all-MiniLM-L6-v2** (sentence-transformers) as the frozen text encoder for turn embeddings.
- **Who&When**, a second benchmark of failed multi-agent runs, used through the authors' processed file (184 records here; the paper says 127, which is unexplained).
- **Kaggle** free GPUs (T4) for the two jobs that need a GPU.

## 4. The data in one paragraph

Each record is one conversation (a "trajectory"): a query, then a list of turns, each turn being `(step, agent name, text)`.
The label is a set of `(agent, error mode)` pairs. There are six frameworks (agentverse, dylan, llm_debate, macnet,
magentic_one, smoagents). The released split is 7,146 train / 1,787 validation / 600 test. No communication edges between
agents are stored, so any graph we build is our own modelling choice.

## 5. Vocabulary

- **Turn**: one message from one agent.
- **Pair-level / agent-level / error-level scores**: the same prediction scored three ways: exact `(agent, mode)` pair,
  agent name only, or error mode only.
- **Micro-F1**: pools all predictions together, so common classes dominate. **Macro-F1**: averages per-class F1, so rare classes count equally.
- **Avg12**: the paper's headline number, the mean of 12 scores (3 levels x micro/macro x 2 benchmarks).
- **Paper split**: the released train/val/test files.
- **Grouped split**: our fix for query leakage. 588 of the 600 test questions also appear in the training data, so a model can
  memorise questions. The grouped split keeps the test set unchanged, removes from train/val every record whose question is in
  test, and re-splits the rest by question (3,865 train / 940 val). Claims about generalisation are made on this split.
- **Trivial floor**: the score of "predict every agent that appears, with all 14 modes". Because about 54-56% of agents are faulty,
  this already scores about 70 on agent-level micro-F1 and about 28 on error-level micro-F1. Several published rows sit below it,
  so every table here includes the floor.
- **Frequency baseline**: always predict the most common pairs for that framework, ignoring the text.
- **Flat MLP**: a small neural network on each agent's averaged turn embeddings, with no graph.
- **0-layer control**: the graph model with message passing switched off (each turn is projected, then averaged). It separates
  "the graph helped" from "the extra projection helped".
- **GNN**: graph neural network. Here: one node per turn plus one query node, edges between consecutive turns, between
  consecutive turns of the same agent, and between the query and every turn.
- **Ledger**: `aegis/results/metrics_history.jsonl`, one JSON line per run (phase, model, seed, split, metrics, git commit).

## 6. Rules the work follows

- Tune only on validation. The 600-record test set is used only for reproducing published rows and for final reporting.
- Never use label-leaking fields as model inputs: `is_injected`, `ground_truth.*`, `metadata.num_injected_agents`,
  `metadata.num_agents`, `injection_strategy`. The loader reads only a whitelist of fields.
- Keep every log. Kaggle logs are pulled into the repo as soon as a kernel finishes.
- Secrets (the Kaggle token and the credential files in the repo root) are never written to a file or committed. They are git-ignored.
- Ask before spending money, downloading more than about 5 GB, or committing data files.

## 7. Where things are

```
aegis/
  src/            schema, loaders, metrics, embedding cache, features, graph builder
    baselines/    random, frequency, flat MLP, DCL-style, top-k decoding, relational probe
    gnn/          turn_gnn.py (the graph model)
    eval/         zero-shot scorer, final_test.py (held-out scoring)
  scripts/        the two Kaggle GPU jobs (zero-shot Qwen, MiniLM embeddings)
  docs/           dataset_audit, decisions (D1-D10), reproduction_log, baselines_log, error_analysis
  results/        metrics_history.jsonl, runs/ (training logs), kaggle_logs/ (saved GPU job outputs)
  tests/          16 unit tests
new-plan.md       the plan this work follows
current-progress.md   what has been done, with all results
```
