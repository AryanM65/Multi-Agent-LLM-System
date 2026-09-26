# Documentation Index

All project documentation, organized by topic. For the project overview, research motivation, and quick-start, see the [root README](../README.md) — this index is for everything beyond that.

## Dataset
- [`dataset/dataset_description.md`](dataset/dataset_description.md) — Full field-by-field schema of `dataset/trials.jsonl`: what's input, what's ground truth, what's bookkeeping-only, plus a worked example mapping one raw record to a GNN training graph.

## Model (GNN fault localization)
- [`model/model.md`](model/model.md) — Deep background: plain-English walkthrough of every concept, the full feature-engineering rationale, and the known dataset caveats (coverage gaps, missing-value handling) that shape the model code.
- [`model/model-plan.md`](model/model-plan.md) — The practical, step-by-step build plan with runnable code snippets, matching what's actually implemented in `../model/`.
- [`model/futurework.md`](model/futurework.md) — Roadmap for after the first trained model: multi-node/multi-fault trials, richer uncertainty signals, dataset expansion, architecture ablations, evaluation depth.

## Infrastructure
- [`infra/kaggle-and-lightning-setup.md`](infra/kaggle-and-lightning-setup.md) — How to actually run dataset generation on Kaggle and Lightning AI: authentication, code-delivery patterns, every practice and error resolved across multiple real generation runs (path gotchas, log-streaming, the NumPy 1.x/2.x ABI cascade, concurrent-GPU-capacity limits). Read this before launching any new generation run.

## Pipeline internals
- [`pipeline/architecture.md`](pipeline/architecture.md) — API-level reference for `src/`: the topology engine, the three uncertainty metrics, the fault-injection taxonomy, and the invariants enforced throughout.
- [`pipeline/local-dev-guide.md`](pipeline/local-dev-guide.md) — Running the pipeline locally (mock mode or a real local model) for sanity checks and small experiments, phase by phase.
- [`pipeline/standalone-agents.md`](pipeline/standalone-agents.md) — Running any single agent (Retriever/Reasoner/Writer) independently via `src/pipelines/` and `scripts/run_agent.py`, useful for isolating which agent is at fault when debugging.

## Results
- [`results/study1-local-results.md`](results/study1-local-results.md) — The original local validation run (chain topology, MLX backend) that established the core methodology, kept for historical reference. The actual training dataset is the much larger multi-topology run described in `dataset/dataset_description.md`.
