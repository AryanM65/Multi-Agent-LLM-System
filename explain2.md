# Branch Explanation — `feature/separate-agent-pipelines`

> **What is this file?**  
> This document explains everything that was added in the Git branch
> `feature/separate-agent-pipelines` (commit `b835158`) on top of the
> original `main` branch (commit `b0e5854`).  
> Every technical term is explained in simple, everyday language.

---

## Quick Summary (TL;DR)

In `main`, the project had one big monolithic pipeline — a single function
that ran all three AI agents (Retriever → Reasoner → Writer) together in
one shot. You could not run any agent by itself.

In `feature/separate-agent-pipelines`, **each agent was given its own
self-contained Python module** that can be run independently, tested on its
own, or chained together in any order. A unified command-line tool was also
added so you can drive any single agent or the full chain from your terminal
with one command.

**Zero existing files were changed.** Everything new was purely additive.

---

## Table of Contents

1. [What "main" Already Had](#1-what-main-already-had)
2. [The Problem This Branch Solves](#2-the-problem-this-branch-solves)
3. [Exact Files Added — At a Glance](#3-exact-files-added--at-a-glance)
4. [Core Concept: How the Agents Work (Self-Consistency Sampling)](#4-core-concept-how-the-agents-work-self-consistency-sampling)
5. [New File 1 — `src/pipelines/__init__.py`](#5-new-file-1--srcpipelinesinitpy)
6. [New File 2 — `src/pipelines/retriever_pipeline.py`](#6-new-file-2--srcpipelinesretriever_pipelinepy)
7. [New File 3 — `src/pipelines/reasoner_pipeline.py`](#7-new-file-3--srcpipelinesreasoner_pipelinepy)
8. [New File 4 — `src/pipelines/writer_pipeline.py`](#8-new-file-4--srcpipelineswriter_pipelinepy)
9. [New File 5 — `scripts/run_agent.py`](#9-new-file-5--scriptsrun_agentpy)
10. [New Files 6-8 — Sample Output JSONL Files](#10-new-files-6-8--sample-output-jsonl-files)
11. [How the Three Pipelines Connect End-to-End](#11-how-the-three-pipelines-connect-end-to-end)
12. [What Was NOT Changed (Backward Compatibility)](#12-what-was-not-changed-backward-compatibility)
13. [Glossary — Every Complex Term Explained Simply](#13-glossary--every-complex-term-explained-simply)

---

## 1. What "main" Already Had

Before this branch, the project had a solid foundation in `main`:

| File | What it does |
|---|---|
| `src/config.py` | Central settings file — model name, temperature, token limits, uncertainty threshold |
| `src/nodes.py` | The engine — `sample_node()` runs an AI agent k times and measures uncertainty |
| `src/pipeline.py` | One function `run_pipeline()` that chains all 3 agents together, no isolation possible |
| `src/faults.py` | Injects deliberate faults (noise / contamination / ceiling) for experiments |
| `src/diagnose.py` | Figures out what kind of fault occurred by retrying with different inputs |
| `scripts/run_study1.py` | Runs the full research experiment over HotpotQA questions |
| `scripts/run_local_debug.py` | Local debug runner |
| `scripts/download_data.py` | Downloads the HotpotQA dataset |

The key limitation: `src/pipeline.py`'s `run_pipeline()` always ran all
three agents at once. There was no way to:

- Run just the Retriever on some text.
- Test just the Writer with a hand-written reasoning chain.
- Save Retriever output and feed it into the Reasoner on a different day.
- Import a single agent into a notebook.

---

## 2. The Problem This Branch Solves

Imagine you are debugging the system. The final answer is wrong. Is the
Retriever choosing the wrong sentences? Or is the Reasoner reasoning
incorrectly? Or is the Writer mangling a correct reasoning chain?

With `main` you had to re-run the entire pipeline every time and manually
inspect intermediate text. There was no clean interface for each agent.

This branch solves that by giving every agent:

1. **Its own Python module** with a clean, typed `run_*_pipeline()` function.
2. **Its own command-line interface (CLI)** so you can run it like a tool
   from your terminal.
3. **Its own output file** (`retriever_output.jsonl`, etc.) so you can
   save and inspect results independently.
4. **The ability to read the previous agent's saved output** — so you can
   chain them together across separate terminal sessions.

---

## 3. Exact Files Added — At a Glance

```
8 files added, 1147 lines of new code, 0 files modified.

btp-pipeline/
├── src/
│   └── pipelines/                        ← ENTIRE NEW PACKAGE
│       ├── __init__.py                   ←  27 lines  — convenience imports
│       ├── retriever_pipeline.py         ← 254 lines  — standalone Retriever
│       ├── reasoner_pipeline.py          ← 277 lines  — standalone Reasoner
│       └── writer_pipeline.py            ← 279 lines  — standalone Writer
├── scripts/
│   └── run_agent.py                      ← 307 lines  — unified CLI
└── results/
    ├── retriever_output.jsonl            ←   1 line   — first test output
    ├── reasoner_output.jsonl             ←   1 line   — first test output
    └── writer_output.jsonl               ←   1 line   — first test output
```

---

## 4. Core Concept: How the Agents Work (Self-Consistency Sampling)

Before reading about each new file, you need to understand **one central
idea** that all three pipeline modules rely on. It comes from `src/nodes.py`
(which already existed in `main`) and is called **self-consistency sampling**.

### The Basic Idea in Plain English

Instead of asking the AI model a question just once, we ask it **the exact
same question k times** (default k = 3). Because the model has some built-in
randomness (controlled by a setting called `temperature`), the answers may
differ slightly each time.

We then look at how much the answers agree with each other:

- **All 3 answers identical → the model is very confident. Uncertainty = 0.0**
- **All 3 answers completely different → the model is confused. Uncertainty ≈ 1.0**
- **2 out of 3 match → Uncertainty = 0.333**
- **1 out of 3 is the top answer → Uncertainty = 0.667**

The formula is:

```
uncertainty = 1  −  (number of times the most common answer appeared / k)
```

### What is "Temperature"?

Temperature is a dial on the AI model that controls how random its outputs
are.

- `temperature = 0` → The model always picks the single most likely next
  word. Deterministic — same input always gives same output.
- `temperature = 0.7` → A little randomness. Mostly sensible answers but
  with natural variation. This is the **default** used in this project.
- `temperature = 1.2` → More randomness. Used only in the "noise fault"
  experiment to simulate a misbehaving model.

### What is "Normalization" (Cleaning Up Answers Before Comparing)?

Before comparing two AI answers, both are cleaned up using
`hotpotqa_normalize()`:

1. Convert to lowercase.
2. Remove all punctuation (`.` `,` `!` etc.).
3. Remove articles (`a`, `an`, `the`).
4. Collapse extra spaces.

So `"The Australian."` and `"australian"` are treated as the same answer.
This prevents false disagreements just because of formatting.

### What Does `sample_node()` Return?

`sample_node()` is the function in `src/nodes.py` that does all of this.
It returns a `NodeResult` object containing:

```
NodeResult:
  node_name   → "retriever", "reasoner", or "writer"
  output      → The best answer (the one the most samples agreed on)
  uncertainty → A number from 0.0 to 1.0
  samples     → The list of all k raw answers from the model
```

Every new pipeline module in this branch calls `sample_node()` internally.

### What is the Uncertainty Threshold?

`UNCERTAINTY_THRESHOLD = 0.75`

If any agent's uncertainty goes above 0.75, it means the model is behaving
more inconsistently than it normally would, and the diagnostic system
(`diagnose.py`) is triggered to figure out why.

The Reasoner's natural baseline is ~0.667 (even on correct answers) because
it writes long step-by-step text that varies in phrasing across samples.
The threshold of 0.75 is set just above that baseline so we only flag
*worse-than-normal* inconsistency.

---

## 5. New File 1 — `src/pipelines/__init__.py`

**Size:** 27 lines  
**Purpose:** Makes `src/pipelines/` a proper Python package and provides
convenient shortcut imports.

### What It Contains

```python
from src.pipelines.retriever_pipeline import RetrieverInput, RetrieverOutput, run_retriever_pipeline
from src.pipelines.reasoner_pipeline  import ReasonerInput,  ReasonerOutput,  run_reasoner_pipeline
from src.pipelines.writer_pipeline    import WriterInput,    WriterOutput,    run_writer_pipeline
```

### Why Does This Matter?

Without this file, Python would not recognise `src/pipelines/` as a package
and you would have to type the full long import path every time. With it,
any other code can just write:

```python
from src.pipelines import run_retriever_pipeline
from src.pipelines import run_reasoner_pipeline
from src.pipelines import run_writer_pipeline
```

It also documents that the MLX model (the AI model loaded into memory) is
loaded **once** as a module-level singleton — meaning the model is loaded
into RAM just one time no matter how many of the three pipelines you import.
Loading a large 8B-parameter model takes about 3 seconds; loading it once
instead of three times saves significant time.

> **What is a "singleton"?**  
> A singleton is a programming pattern where a resource (like a loaded AI
> model) is created only once and then reused. Like turning on your computer
> once and then opening many apps, rather than restarting the computer each
> time you open a new app.

---

## 6. New File 2 — `src/pipelines/retriever_pipeline.py`

**Size:** 254 lines  
**Role of the Retriever:** Given a question and a pile of paragraphs, pick
out **only the sentences that are directly relevant** to answering the
question. Throw away everything else.

### Real-World Analogy

Imagine a research assistant who reads through 10 newspaper articles and
highlights only the sentences that are actually useful for answering your
question. The Retriever does exactly that for AI.

### The Input

```
question: "What nationality is the director of Crocodile Dundee?"

context:  "[Crocodile Dundee] Crocodile Dundee is a 1986 Australian comedy
           film directed by Peter Faiman. It was produced by John Cornell...
           [Peter Faiman] Peter Faiman is an Australian television director.
           He is best known for directing the 1986 film Crocodile Dundee...
           [Paul Hogan] Paul Hogan is an Australian actor and comedian..."
```

The context is a string of multiple paragraphs — some directly relevant,
others just noise (called *distractors*). The Retriever must separate the
useful from the useless.

### The Prompt Sent to the Model

```
You are a Retriever agent. Given the question and candidate paragraphs,
select and return only the sentences that are directly relevant to answering
the question. Do not add any commentary or explanation — only return the
relevant sentences.

Question: What nationality is the director of Crocodile Dundee?
Paragraphs:
[full context text here]

Relevant sentences:
```

The prompt ends with `"Relevant sentences:"` — this is a deliberate trick.
The model is trained to complete text, so it will continue from where the
prompt leaves off and output only the selected sentences.

### What Are `RetrieverInput` and `RetrieverOutput`?

These are **dataclasses** — Python structures that hold named pieces of data.

> **What is a dataclass?**  
> A dataclass is like a labelled box. Instead of shoving values into a plain
> list or dictionary (where you have to remember which position means what),
> a dataclass gives every value a clear name. It makes code much easier to
> read and prevents mistakes.

```
RetrieverInput:
  question  → the question string
  context   → all the raw paragraphs as one big string

RetrieverOutput:
  question          → same question (passed along so downstream agents can see it)
  selected_evidence → the filtered sentences the Retriever chose
  result            → the full NodeResult (uncertainty score + all k raw samples)
```

### The `run_retriever_pipeline()` Function (the Main Entry Point)

```python
def run_retriever_pipeline(question, context, k=3, temperature=0.7):
    prompt = build_prompt(question, context)
    result = sample_node("retriever", prompt, k=k, temperature=temperature)
    return RetrieverOutput(
        question=question,
        selected_evidence=result.output,   # best answer
        result=result,                     # full NodeResult with uncertainty
    )
```

Step by step:
1. Build the prompt string (question + context).
2. Call `sample_node()` which asks the model k=3 times and measures uncertainty.
3. Wrap the result in a clean `RetrieverOutput` dataclass.
4. Return it so the caller can use `output.selected_evidence` and `output.result.uncertainty`.

### The `to_dict()` Method and JSONL Saving

Each output dataclass has a `to_dict()` method that converts it to a plain
Python dictionary, which is then written to a JSONL file:

```
results/retriever_output.jsonl
```

> **What is JSONL?**  
> JSONL stands for "JSON Lines". It's a text file where each line is a
> complete, self-contained JSON object. This is great for logging because
> you can just append a new line for each run without ever reading or
> rewriting the whole file.

### The Command-Line Interface (CLI)

Every pipeline module is also a standalone CLI tool. You run it directly
from your terminal:

```bash
# From inside btp-pipeline/

# Give context as a string
python -m src.pipelines.retriever_pipeline \
    --question "What nationality is the director of Crocodile Dundee?" \
    --context  "[Crocodile Dundee] Australian film directed by Peter Faiman. Peter Faiman is an Australian television director."

# Or load context from a file
python -m src.pipelines.retriever_pipeline \
    --question "What nationality is the director of Crocodile Dundee?" \
    --context-file /path/to/my_context.txt

# Extra options:
#   --k 5          ask the model 5 times instead of 3
#   --temperature 0.9   make the model slightly more random
#   --no-save          don't write to retriever_output.jsonl
```

### Terminal Output Example

```
[retriever_pipeline] Question : What nationality is the director of Crocodile Dundee?
[retriever_pipeline] Context  : [Crocodile Dundee] Australian film directed by Peter Faiman...
[retriever_pipeline] k=3  temperature=0.7
[retriever_pipeline] Running Retriever agent...

────────────────────────────────────────────────────────────────
  [RETRIEVER RESULT]
  Uncertainty      : 0.0000  (0=certain, 1=random)
  Selected evidence: Peter Faiman is an Australian television director.

  All 3 samples:
    [0] Peter Faiman is an Australian television director.
    [1] Peter Faiman is an Australian television director.
    [2] Peter Faiman is an Australian television director.
────────────────────────────────────────────────────────────────

[retriever_pipeline] Output saved → results/retriever_output.jsonl
```

---

## 7. New File 3 — `src/pipelines/reasoner_pipeline.py`

**Size:** 277 lines  
**Role of the Reasoner:** Given the question and the filtered evidence from
the Retriever, produce a **full step-by-step chain-of-thought reasoning
trace** — show all the logical steps from evidence to conclusion.

### Real-World Analogy

Imagine a detective who has just received the highlighted sentences from the
research assistant (Retriever). The detective now writes out their full
reasoning: "Given that Person X directed this film, and Person X is
Australian, therefore the director of this film is Australian." That's the
Reasoner.

### What Is "Chain-of-Thought"?

Chain-of-thought (CoT) is a technique where you tell the AI to "think out
loud" and write down each reasoning step rather than jumping straight to an
answer. It produces outputs like:

```
1. Who directed Crocodile Dundee?
   → The evidence says "Peter Faiman directed the film."
2. What nationality is Peter Faiman?
   → The evidence says "Peter Faiman is an Australian television director."
3. Connecting the dots:
   → The director of Crocodile Dundee is Peter Faiman.
   → Peter Faiman is Australian.
   → Therefore the director is Australian.
Answer: Australian.
```

Chain-of-thought makes the model's reasoning transparent and tends to
produce more accurate answers for multi-step questions.

### Why Does the Reasoner Get a Bigger Token Budget?

```
MAX_TOKENS = 128           (Retriever and Writer)
MAX_TOKENS_REASONER = 200  (Reasoner)
```

> **What are "tokens"?**  
> Tokens are the pieces that AI models split text into — roughly, one token
> ≈ 0.75 words. So 200 tokens ≈ ~150 words. The Reasoner needs more words
> because it is writing out multiple reasoning steps.

The `sample_node()` function in `nodes.py` already handles this
automatically: it checks if `node_name == "reasoner"` and uses
`MAX_TOKENS_REASONER` instead of `MAX_TOKENS`.

### The Input

```
question: "What nationality is the director of Crocodile Dundee?"
evidence: "Peter Faiman is an Australian television director."
          ↑ This comes directly from RetrieverOutput.selected_evidence
```

### The Prompt Sent to the Model

```
You are a Reasoning agent. Given the evidence below, reason step by step
to derive the answer to the question. Show your full chain of thought.

Question: What nationality is the director of Crocodile Dundee?
Evidence: Peter Faiman is an Australian television director.

Reasoning:
```

Again the prompt ends with `"Reasoning:"` — the model continues from there
and writes out its step-by-step logic.

### `ReasonerInput` and `ReasonerOutput`

```
ReasonerInput:
  question  → the question string
  evidence  → the selected_evidence from the Retriever

ReasonerOutput:
  question        → echoed
  evidence        → echoed (for traceability — so you can see exactly what was fed in)
  reasoning_chain → the full chain-of-thought text
  result          → NodeResult (uncertainty, samples)
```

> **What does "echoed" mean?**  
> The output carries copies of the input fields (question, evidence) so
> that anyone reading the output file can see exactly what went in, without
> needing to look at a separate file. It makes the output self-contained
> and traceable.

### Important Note: Why Is Reasoner Uncertainty Higher?

The Reasoner uncertainty is **expected to be ~0.667 even on completely
correct answers**. Here is why:

The Reasoner writes long paragraphs of text. Even when all 3 samples reach
the same conclusion ("Australian"), they express it in different words:

- Sample 1: "...Therefore, the nationality is **Australian**."
- Sample 2: "...We conclude that he is **of Australian nationality**."
- Sample 3: "...The director is **Australian**."

After normalization, `"australian"`, `"of australian nationality"`, and
`"australian"` — samples 1 and 3 match, sample 2 doesn't. Result: 2/3
agreement → uncertainty = 0.333. But in more complex cases all 3 may differ,
giving uncertainty = 0.667.

This is **not a bug** — it's an inherent property of free-form text
generation. The threshold of 0.75 is set above this natural baseline.

### The `--from-retriever-output` Flag (Pipeline Chaining Feature)

This is one of the most useful additions in this branch. Instead of
manually copy-pasting the Retriever's output, you can tell the Reasoner
to read it from the saved JSONL file:

```bash
# Step 1: Run Retriever, it saves to results/retriever_output.jsonl
python -m src.pipelines.retriever_pipeline \
    --question "What nationality is the director of Crocodile Dundee?" \
    --context "..."

# Step 2: Run Reasoner, reading from the saved Retriever output
python -m src.pipelines.reasoner_pipeline \
    --question "What nationality is the director of Crocodile Dundee?" \
    --from-retriever-output results/retriever_output.jsonl
```

Internally this calls `_load_latest_retriever_output()` which reads the
**last line** of the JSONL file and extracts `selected_evidence` from it.

This is a **mutually exclusive** argument with `--evidence` — you provide
one or the other, not both.

---

## 8. New File 4 — `src/pipelines/writer_pipeline.py`

**Size:** 279 lines  
**Role of the Writer:** Given the question and the Reasoner's step-by-step
reasoning, produce **a single, concise final answer** — no explanation,
no preamble, just the answer.

### Real-World Analogy

The detective (Reasoner) has written out a full detective's report. The
editor (Writer) reads the report and writes a single sentence: "Australian."
That's the Writer.

### The Input

```
question:  "What nationality is the director of Crocodile Dundee?"
reasoning: "To determine the nationality... Peter Faiman directed it...
            The evidence says he is Australian... Therefore Australian.
            Answer: Australian."
           ↑ This comes from ReasonerOutput.reasoning_chain
```

### The Prompt Sent to the Model

```
You are a Writer agent. Given the reasoning trace below, produce a final,
concise answer to the question. Output only the answer — no explanation,
no preamble.

Question: What nationality is the director of Crocodile Dundee?
Reasoning: [full reasoning chain here]

Final answer:
```

The model continues from `"Final answer:"` and outputs just one short answer.

### `WriterInput` and `WriterOutput`

```
WriterInput:
  question  → the question
  reasoning → the reasoning_chain from the Reasoner

WriterOutput:
  question     → echoed
  reasoning    → echoed
  final_answer → "Australian."
  result       → NodeResult (uncertainty, samples)
```

### Why Is Writer Uncertainty Expected to Be Very Low?

Short factual answers like "Australian." converge immediately. Even if the
Reasoner wrote slightly different reasoning chains, the Writer reading any
of them will almost always produce the same one-word answer. So:

- **Clean input → uncertainty ≈ 0.0** (normal behaviour)
- **High Writer uncertainty** after clean reasoning → genuine problem: the
  question is ambiguous, or the Reasoner gave corrupted/contradictory reasoning.

### The `--from-reasoner-output` Flag

Same pattern as the Reasoner's `--from-retriever-output`:

```bash
python -m src.pipelines.writer_pipeline \
    --question "What nationality is the director of Crocodile Dundee?" \
    --from-reasoner-output results/reasoner_output.jsonl
```

Reads the last line of `reasoner_output.jsonl` and extracts `reasoning_chain`.

---

## 9. New File 5 — `scripts/run_agent.py`

**Size:** 307 lines  
**Purpose:** A single unified command-line tool that acts as a **router** —
you tell it which agent you want and it dispatches to the correct module.

### The Four Modes

```bash
python scripts/run_agent.py [MODE] [OPTIONS]

# MODE can be one of:
#   retriever   → runs only the Retriever
#   reasoner    → runs only the Reasoner
#   writer      → runs only the Writer
#   all         → chains all three sequentially, prints uncertainty table
```

### Mode: `retriever`

```bash
python scripts/run_agent.py retriever \
    --question "What nationality is the director of Crocodile Dundee?" \
    --context "Peter Faiman is an Australian television director. ..."
```

Internally this calls `retriever_pipeline.main()` — exactly the same as
running `python -m src.pipelines.retriever_pipeline`.

### Mode: `reasoner`

```bash
# With inline evidence:
python scripts/run_agent.py reasoner \
    --question "..." \
    --evidence "Peter Faiman is an Australian television director."

# Or with saved Retriever output:
python scripts/run_agent.py reasoner \
    --question "..." \
    --from-retriever-output results/retriever_output.jsonl
```

### Mode: `writer`

```bash
# With inline reasoning:
python scripts/run_agent.py writer \
    --question "..." \
    --reasoning "Peter Faiman directed it. He is Australian. Therefore Australian."

# Or with saved Reasoner output:
python scripts/run_agent.py writer \
    --question "..." \
    --from-reasoner-output results/reasoner_output.jsonl
```

### Mode: `all` (The Most Useful Mode)

This chains all three agents sequentially in memory and prints a
**per-node uncertainty comparison table** at the end:

```bash
python scripts/run_agent.py all \
    --question "What nationality is the director of Crocodile Dundee?" \
    --context "..."
```

What happens internally:

```python
# 1. Run Retriever
ret_out = run_retriever_pipeline(question, context, k, temperature)

# 2. Feed Retriever output into Reasoner (directly in memory, no file needed)
rea_out = run_reasoner_pipeline(question, ret_out.selected_evidence, k, temperature)

# 3. Feed Reasoner output into Writer
wri_out = run_writer_pipeline(question, rea_out.reasoning_chain, k, temperature)

# 4. Save all three outputs to their respective JSONL files
# 5. Print the uncertainty table
```

### The Uncertainty Summary Table (Printed at the End of `all` Mode)

```
════════════════════════════════════════════════════════════════════════
  FULL PIPELINE: Retriever → Reasoner → Writer
  Question   : What nationality is the director of Crocodile Dundee?
  k=3  temperature=0.7
════════════════════════════════════════════════════════════════════════

[1/3] Running RETRIEVER...
  ✓ Retriever  uncertainty = 0.0000
  Selected evidence: Peter Faiman is an Australian television director.

[2/3] Running REASONER...
  ✓ Reasoner   uncertainty = 0.6667
  Reasoning (first 300 chars): To determine the nationality of the director...

[3/3] Running WRITER...
  ✓ Writer     uncertainty = 0.0000
  Final answer: Australian.

════════════════════════════════════════════════════════════════════════
  PER-NODE UNCERTAINTY SUMMARY  (never collapsed to a scalar)
  Node            Uncertainty  Status
  ────────────────────────────────────────────
  retriever          0.0000  ✓  ok
  reasoner           0.6667  ✓  ok
  writer             0.0000  ✓  ok

  Threshold (UNCERTAINTY_THRESHOLD): 0.75
  Final answer : Australian.
════════════════════════════════════════════════════════════════════════
```

Notice the comment `"never collapsed to a scalar"` — this is an important
design rule enforced throughout the project.

> **What does "never collapsed to a scalar" mean?**  
> A scalar is a single number. The design rule says you must **never** add
> up or average the uncertainties from different nodes into one combined
> number. Each agent's uncertainty must always be kept and displayed
> separately. This is because collapsing them would hide which specific
> agent is misbehaving — the whole point of the system is to pinpoint
> *which* agent has a problem.

### The Common Options (Available in All Modes)

| Option | Default | What it does |
|---|---|---|
| `--k 5` | 3 | Ask the model 5 times instead of 3 for better uncertainty estimates |
| `--temperature 0.9` | 0.7 | Make the model slightly more random |
| `--no-save` | (saves by default) | Skip writing to `results/*.jsonl` |

### How the Internal Dispatch Works

```python
dispatch = {
    "retriever": run_retriever_mode,
    "reasoner":  run_reasoner_mode,
    "writer":    run_writer_mode,
    "all":       run_all_mode,
}
dispatch[args.agent](args)
```

This is a clean Python dictionary acting as a switch statement. Based on
what you typed as the first argument, the corresponding function is called.

---

## 10. New Files 6-8 — Sample Output JSONL Files

Three JSONL files were added under `btp-pipeline/results/` containing the
output from the **first successful test run** of the new pipeline modules.
They serve as living documentation of what real output looks like, and as
a baseline for future comparisons.

### `results/retriever_output.jsonl`

```json
{
  "node": "retriever",
  "question": "What nationality is the director of Crocodile Dundee?",
  "selected_evidence": "Peter Faiman is an Australian television director.",
  "uncertainty": 0.0,
  "samples": [
    "Peter Faiman is an Australian television director.",
    "Peter Faiman is an Australian television director.",
    "Peter Faiman is an Australian television director."
  ]
}
```

**Reading this:**  
- `"node": "retriever"` — this is Retriever output.
- `"selected_evidence"` — the sentence the model chose as most relevant.
- `"uncertainty": 0.0` — all 3 samples were identical. The model is 100% confident.
- `"samples"` — all 3 identical answers.

### `results/reasoner_output.jsonl`

```json
{
  "node": "reasoner",
  "question": "What nationality is the director of Crocodile Dundee?",
  "evidence": "Peter Faiman is an Australian television director.",
  "reasoning_chain": "To determine the nationality of the director of
    Crocodile Dundee, we need to follow a logical sequence...\n
    1. Identify the director: Peter Faiman.\n
    2. Use the evidence: 'Peter Faiman is an Australian television director.'\n
    3. Conclusion: The nationality is Australian.\nAnswer: Australian.",
  "uncertainty": 0.6667,
  "samples": [
    "(long version 1 — reaches Australian)",
    "(long version 2 — different phrasing, reaches Australian)",
    "(long version 3 — yet another phrasing, reaches Australian)"
  ]
}
```

**Reading this:**  
- `"uncertainty": 0.6667` — 2 out of 3 samples matched after normalization
  (the third was phrased differently). This is the **expected baseline** for
  the Reasoner, not a problem.
- `"reasoning_chain"` — the best sample (the one the majority voted for)
  is stored here and passed to the Writer.

### `results/writer_output.jsonl`

```json
{
  "node": "writer",
  "question": "What nationality is the director of Crocodile Dundee?",
  "reasoning": "[full Reasoner chain here]",
  "final_answer": "Australian.",
  "uncertainty": 0.0,
  "samples": ["Australian.", "Australian", "Australian."]
}
```

**Reading this:**  
- `"uncertainty": 0.0` — after normalization, `"Australian."` and `"Australian"`
  both become `"australian"`. All 3 match. Perfectly self-consistent.
- `"final_answer": "Australian."` — this is the final pipeline output.

### Overall Test Result: All Nodes Below Threshold ✓

| Node | Uncertainty | Threshold | Status |
|---|---|---|---|
| Retriever | 0.0000 | 0.75 | ✓ ok |
| Reasoner | 0.6667 | 0.75 | ✓ ok (expected) |
| Writer | 0.0000 | 0.75 | ✓ ok |

The system correctly answered "Australian" with all agents well within
their expected uncertainty ranges. No diagnostic protocol was triggered.

The model used: `mlx-community/Qwen3-8B-4bit` running locally via `mlx_lm`
on Apple Silicon.

---

## 11. How the Three Pipelines Connect End-to-End

```
                    USER INPUT
                        │
            ┌───────────▼────────────┐
            │  Question + Context    │
            └───────────┬────────────┘
                        │
        ╔═══════════════▼═══════════════╗
        ║         RETRIEVER             ║
        ║  "Pick only useful sentences" ║
        ║  Runs k=3 times               ║
        ║  Uncertainty = 0.0            ║
        ║  Output: selected_evidence    ║
        ╚═══════════════╤═══════════════╝
                        │  selected_evidence
                        │  ("Peter Faiman is an Australian television director.")
        ╔═══════════════▼═══════════════╗
        ║          REASONER             ║
        ║  "Think step by step"         ║
        ║  Runs k=3 times               ║
        ║  Uncertainty = 0.6667         ║
        ║  Output: reasoning_chain      ║
        ╚═══════════════╤═══════════════╝
                        │  reasoning_chain
                        │  ("Peter Faiman directed it... He is Australian... Therefore Australian.")
        ╔═══════════════▼═══════════════╗
        ║           WRITER              ║
        ║  "Write one concise answer"   ║
        ║  Runs k=3 times               ║
        ║  Uncertainty = 0.0            ║
        ║  Output: final_answer         ║
        ╚═══════════════╤═══════════════╝
                        │
            ┌───────────▼────────────┐
            │    "Australian."       │
            └────────────────────────┘
```

### The Three Ways to Run This Chain

#### Method A: Pure In-Memory (Fastest)

```python
from src.pipelines import run_retriever_pipeline, run_reasoner_pipeline, run_writer_pipeline

ret = run_retriever_pipeline(question=q, context=ctx)
rea = run_reasoner_pipeline(question=q, evidence=ret.selected_evidence)
wri = run_writer_pipeline(question=q, reasoning=rea.reasoning_chain)

print(wri.final_answer)
print(f"Retriever uncertainty: {ret.result.uncertainty}")
print(f"Reasoner  uncertainty: {rea.result.uncertainty}")
print(f"Writer    uncertainty: {wri.result.uncertainty}")
```

#### Method B: Saved Files (Run Steps Separately)

```bash
# Run Retriever → saves results/retriever_output.jsonl
python -m src.pipelines.retriever_pipeline --question "..." --context "..."

# Run Reasoner → reads retriever_output.jsonl → saves reasoner_output.jsonl
python -m src.pipelines.reasoner_pipeline \
    --question "..." --from-retriever-output results/retriever_output.jsonl

# Run Writer → reads reasoner_output.jsonl → saves writer_output.jsonl
python -m src.pipelines.writer_pipeline \
    --question "..." --from-reasoner-output results/reasoner_output.jsonl
```

#### Method C: Unified CLI (Simplest)

```bash
python scripts/run_agent.py all --question "..." --context "..."
```

---

## 12. What Was NOT Changed (Backward Compatibility)

One of the explicit design goals of this branch was:
**"Zero changes to existing pipeline.py / run_study1.py (backward compat)"**

This means:

| Existing File | Changed? | Notes |
|---|---|---|
| `src/config.py` | ❌ No | All settings unchanged |
| `src/nodes.py` | ❌ No | `sample_node()`, `NodeResult`, `PipelineTrace` untouched |
| `src/pipeline.py` | ❌ No | The monolithic `run_pipeline()` still works exactly as before |
| `src/faults.py` | ❌ No | All fault injectors untouched |
| `src/diagnose.py` | ❌ No | Full diagnostic protocol untouched |
| `scripts/run_study1.py` | ❌ No | Main experiment script untouched |
| `scripts/run_local_debug.py` | ❌ No | Untouched |
| `scripts/download_data.py` | ❌ No | Untouched |

Anyone using the original `run_study1.py` or `run_pipeline()` function
directly will see no difference at all. The new `src/pipelines/` package
is purely additive — it adds new capabilities without touching anything
that already existed.

> **Why is backward compatibility important?**  
> In a research project, if you change the pipeline while an experiment is
> running (or between experiment runs), the results become incomparable —
> you can't tell if a change in results came from your code changes or from
> natural variation. Keeping old code untouched means all previous
> experiment data remains valid and comparable.

---

## 13. Glossary — Every Complex Term Explained Simply

| Term | Simple Explanation |
|---|---|
| **Agent** | An AI model that has been given a specific role and a specific job to do (Retriever, Reasoner, or Writer). |
| **Pipeline** | A series of AI agents where each one's output is fed as input to the next. Like an assembly line. |
| **Node** | One step / one agent in the pipeline. |
| **Self-consistency sampling** | Asking the AI the exact same question multiple times and checking if it agrees with itself. |
| **k** | How many times you ask the AI (default: 3). More k = better uncertainty estimate but slower. |
| **Temperature** | How random the AI model is. 0 = always the same answer, 2 = very unpredictable. Default is 0.7. |
| **Uncertainty** | A number from 0 to 1 measuring how inconsistent the AI's answers were. 0 = perfectly consistent, 1 = completely inconsistent. |
| **Agreement rate** | What fraction of the k samples matched the most popular answer. Uncertainty = 1 − agreement rate. |
| **Modal answer** | The answer that appeared the most times across k samples. This is the "best" answer. |
| **Normalization** | Cleaning up text (lowercase, remove punctuation, remove articles) so `"The Australian."` and `"australian"` count as the same answer. |
| **Chain-of-thought (CoT)** | Writing out every reasoning step explicitly, like showing your work in a maths exam. Makes AI reasoning transparent and often more accurate. |
| **HotpotQA** | A research dataset of tricky questions that require reasoning over two separate pieces of information (multi-hop). |
| **Distractor paragraphs** | Irrelevant paragraphs included alongside the real useful ones, designed to trick the Retriever into choosing the wrong text. |
| **JSONL (JSON Lines)** | A text file where each line is its own independent JSON object. Good for appending results one-at-a-time without loading the whole file. |
| **Dataclass** | A Python class whose main purpose is to hold named data fields. Like a form with labelled boxes instead of a pile of anonymous values. |
| **Singleton** | A programming pattern where a resource (here: the AI model loaded in RAM) is created only once and reused, instead of being loaded fresh every time it's needed. Saves ~3 seconds per load. |
| **MLX / mlx_lm** | Apple's machine learning framework for running AI models on Apple Silicon chips (M1/M2/M3). Used here instead of Ollama because the model is in MLX format. |
| **Qwen3-8B-4bit** | The specific AI model used. Qwen3 = model family (by Alibaba). 8B = 8 billion parameters (size). 4bit = compressed to 4-bit numbers to save memory and run faster. |
| **Token** | The unit AI models work with. Roughly 1 token ≈ 0.75 words. `MAX_TOKENS = 128` means the model's answer can be at most ~96 words. |
| **MAX_TOKENS_REASONER** | A larger token budget (200) given only to the Reasoner because it writes long step-by-step text that won't fit in 128 tokens. |
| **UNCERTAINTY_THRESHOLD** | The cutoff value (0.75). If any agent's uncertainty exceeds this, the diagnostic system (`diagnose.py`) is triggered to find out why. |
| **Backward compatible** | New code doesn't break or change any existing code. Old scripts still work exactly as before. |
| **Mutually exclusive arguments** | CLI arguments where you can provide one or the other but not both. For example `--evidence` and `--from-retriever-output` — you pick one. |
| **Scalar** | A single number. The design rule says you must NEVER collapse per-node uncertainties into one combined number, because that would hide which specific agent is misbehaving. |
| **Traceability** | Being able to see exactly what input went into each step. Achieved by echoing input fields in the output dataclasses and JSONL files. |
| **Inference Gap** | An experimental metric (in `diagnose.py`, not used in Study 1) that measures how semantically different the output is from the input using sentence embeddings. A high gap might indicate the model "drifted" away from the topic. |
| **sys.path** | Python's list of directories to search when you write `import something`. Each pipeline module adds its own project root to this list so imports like `from src.config import ...` always work, regardless of where you run the script from. |
| **`__main__`** | A special Python name. When a Python file is run directly (not imported), its `__name__` equals `"__main__"`. Each pipeline module has an `if __name__ == "__main__": main()` block that makes it runnable as a CLI tool. |
