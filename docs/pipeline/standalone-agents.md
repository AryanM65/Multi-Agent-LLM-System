# Standalone Single-Agent Pipelines

> `src/pipeline.py`'s `run_pipeline()` always runs Retriever → Reasoner → Writer together. `src/pipelines/` (a separate package) gives each agent its own standalone module — usable independently, testable in isolation, and chainable across separate terminal sessions via saved JSONL files. This is purely additive: nothing in `src/pipeline.py` or the main generation scripts depends on it or is changed by it.

---

## Why this exists

When debugging a wrong final answer, you need to isolate which agent is at fault: is the Retriever choosing bad evidence, the Reasoner reasoning incorrectly, or the Writer mangling correct reasoning? Running the full pipeline every time and manually inspecting intermediate text is slow. Each agent having its own CLI + JSONL output makes this a one-command check.

## The three standalone modules

| Module | Role | Input | Output |
|---|---|---|---|
| `src/pipelines/retriever_pipeline.py` | Given a question + raw paragraphs, select only the sentences directly relevant to answering the question | `question`, `context` | `selected_evidence`, `NodeResult` (uncertainty + k raw samples) |
| `src/pipelines/reasoner_pipeline.py` | Given the question + evidence, produce a full step-by-step chain-of-thought | `question`, `evidence` | `reasoning_chain`, `NodeResult` |
| `src/pipelines/writer_pipeline.py` | Given the question + reasoning, produce one concise final answer | `question`, `reasoning` | `final_answer`, `NodeResult` |

Each module:
1. Is runnable directly (`python -m src.pipelines.retriever_pipeline --question ... --context ...`).
2. Writes its output to `results/{node}_output.jsonl` by default (`--no-save` to skip).
3. Can read the *previous* agent's saved output instead of taking inline text (`--from-retriever-output`, `--from-reasoner-output`), so you can chain agents across separate terminal sessions without copy-pasting.

## Example: chaining via saved files
```bash
python -m src.pipelines.retriever_pipeline \
    --question "What nationality is the director of Crocodile Dundee?" \
    --context "[Crocodile Dundee] ...Peter Faiman... [Peter Faiman] Australian television director..."
# → results/retriever_output.jsonl

python -m src.pipelines.reasoner_pipeline \
    --question "What nationality is the director of Crocodile Dundee?" \
    --from-retriever-output results/retriever_output.jsonl
# → results/reasoner_output.jsonl

python -m src.pipelines.writer_pipeline \
    --question "What nationality is the director of Crocodile Dundee?" \
    --from-reasoner-output results/reasoner_output.jsonl
# → results/writer_output.jsonl
```

## Unified CLI: `scripts/run_agent.py`

A router over the same three modules, plus an `all` mode that chains them in-memory and prints a per-node uncertainty summary table (never collapsed to a scalar — see the core design rule in [`docs/pipeline/architecture.md`](architecture.md)):

```bash
python scripts/run_agent.py retriever --question "..." --context "..."
python scripts/run_agent.py reasoner --question "..." --evidence "..."
python scripts/run_agent.py writer --question "..." --reasoning "..."
python scripts/run_agent.py all --question "..." --context "..." --k 3
python scripts/run_agent.py all --question "..." --context "..." --k 3 --mock   # no model needed
```

`all` mode output:
```
[1/3] RETRIEVER  uncertainty = 0.0000   Selected evidence: Peter Faiman is an Australian television director.
[2/3] REASONER   uncertainty = 0.6667   Reasoning: To determine the nationality... (expected baseline, see local-dev-guide.md)
[3/3] WRITER     uncertainty = 0.0000   Final answer: Australian.
```

## Why Reasoner uncertainty is expected to be higher than Retriever/Writer

The Reasoner writes long free-form chain-of-thought text. Even when all k samples reach the same conclusion, they phrase it differently — after lexical normalization, this still often disagrees. A ~0.667 lexical-uncertainty baseline on clean input is expected here, not a bug (this is exactly what the semantic-uncertainty metric was added to correct for — see `docs/pipeline/architecture.md`).

## Common options (all modes)

| Option | Default | Effect |
|---|---|---|
| `--k N` | 3 | Self-consistency samples per node |
| `--temperature T` | 0.7 | Sampling temperature |
| `--no-save` | (saves by default) | Skip writing to `results/*.jsonl` |
| `--mock` | off | Deterministic mock backend, no model needed |
