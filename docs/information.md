# Project Information — Complete Record

> Full record of everything done on this project to date: the research question, the pipeline and fault-injection framework, dataset generation (both the finished k=5 dataset and the in-progress k=10 regeneration), the full GNN modeling and ablation history, deeper evaluation (ROC/PR/confusion-matrix analysis), all report-writing/documentation produced, every chart generated, the tooling used to produce them, and everything currently open/unresolved. Written 2026-09-29; supersedes the earlier k=5-only version of this file.

---

## 1. What the project is

**Title (working):** *Signature-Aware Orchestration for Uncertainty Propagation in Multi-Agent LLM Systems* — BTech Thesis (BTP), 2026.

**The research question:** Multi-agent LLM pipelines (e.g. Retriever → Reasoner → Writer, or more complex DAGs) can fail silently — an error introduced at one agent propagates downstream, and the final output still looks fluent and confident. Standard confidence-scoring and self-consistency methods only examine the final output, so they cannot say *which* agent actually failed or *why*. This project asks:

> Can per-node self-consistency uncertainty, combined with the pipeline's own topology (which agent feeds which), be used to automatically localize the true source of a fault and classify the fault's type — with no external oracle, no human annotation — and does any such method generalize to pipeline shapes never seen during training?

The project runs in four phases: (1) build the pipeline, uncertainty measurement, and fault-injection framework; (2) generate a large, verified, labeled dataset across many topologies; (3) train and evaluate models — from naive baselines to a Graph Neural Network — on the fault-localization task; (4) regenerate the dataset at a finer self-consistency resolution (k=10 instead of k=5) to push accuracy further. Phases 1–3 are complete (§2–§5 below). Phase 4 is in progress, with partial/preview results (§7).

---

## 2. Phase 1 — Pipeline, uncertainty measurement, and fault injection (complete)

### 2.1 The pipeline

Code: `btp-pipeline/src/`. The pipeline is not hardcoded to a 3-node chain — it is an arbitrary directed acyclic graph (DAG) of nodes, each with a role:

- `NodeSpec(node_id, role, instruction)` — role is one of `retriever` / `reasoner` / `writer`.
- `Topology(nodes, edges, topology_id)` — the graph structure.
- Role ordering is enforced non-decreasing along every edge (`ROLE_RANK = {retriever: 0, reasoner: 1, writer: 2}`) — a retriever can feed a reasoner or writer, a reasoner can feed a reasoner or writer, a writer feeds nothing.
- Execution follows a topological sort (Kahn's algorithm), so any valid DAG — chains, fan-ins, fan-outs, diamonds, deep/mixed multi-layer graphs — executes deterministically.
- `run_pipeline(topo, question, context, k, temperature, fault_config=None) -> PipelineTrace` builds each node's prompt from its parents' outputs, applies any active fault, and samples the node.

### 2.2 Self-consistency uncertainty measurement

Each node is sampled **k times independently** at the same temperature. Three complementary uncertainty metrics are computed, and — this is the single most important structural decision in the whole project — **they are never collapsed into a single scalar**. `PipelineTrace.uncertainties()` always returns `Dict[node_id, float]`, one value per node. This is what makes the problem naturally suited to a graph-structured (GNN) model later: every node in the topology carries its own feature.

| Metric | Formula / method | Computed for |
|---|---|---|
| **Lexical uncertainty** | `1 − (fraction of k samples matching the modal normalized answer)`. Normalization = lowercase → strip punctuation → remove articles → collapse whitespace (matches HotpotQA's own SQuAD-style evaluation). | Every node |
| **Semantic uncertainty** | Entropy over embedding-cluster assignment of the k samples, via `all-MiniLM-L6-v2` sentence embeddings. | `reasoner`, `writer` only |
| **Jaccard uncertainty** | `1 − mean pairwise Jaccard set-overlap`, for outputs that parse into multiple discrete items. | `retriever` only, and only when multi-item |

**Why semantic uncertainty exists at all:** the Reasoner writes free-form chain-of-thought, so even when its *conclusion* is identical across samples, the *wording* varies — lexical matching alone gives a non-zero "baseline" uncertainty (~0.667 empirically) on entirely clean data, which is not itself a fault signal. Semantic clustering separates genuine disagreement from this phrasing variance. This was one of the first real findings of the project (see §2.4).

### 2.3 Fault injection

Code: `btp-pipeline/src/faults.py`. Three fault types, each targetable at **any node** in the topology, dispatch is **role-aware** (keyed on the node's role, not a string match against its name — important because non-chain topologies use custom node IDs like `retriever_a`, not the literal string `"retriever"`):

| Fault | Mechanism | Expected recovery pattern |
|---|---|---|
| `noise` | Sampling temperature raised from 0.7 to 1.2 at the targeted node only; input unchanged. | Recovers on a same-input retry at normal temperature — it's instability, not bad input. |
| `contamination` | Targeted node's input replaced with a plausible-but-wrong distractor (Retriever: gold paragraphs swapped for ranked distractors; Reasoner/Writer: parent's output replaced from a substitute bank, self-exclusion enforced so the "wrong" substitute can never equal the clean value). | Recovers on a clean-input retry, not a same-input retry. |
| `ceiling` | Targeted node's evidence/instructions stripped or hardened to force information loss, with **mandatory verification** that the correct answer no longer survives before the trial is logged — if it does survive, the trial is skipped (never mislabeled). | Does not recover on either retry — a genuine capability gap, not noise or bad context. |

Every fault injector is deterministically seeded via `make_rng(question_id, fault_type, target_node)` — never the global `random` module — so re-running any trial under the same conditions reproduces byte-identical corrupted input. A clean control trial is always run first per question, establishing a per-question baseline that fault trials are later verified against via a z-score deviation check.

### 2.4 Diagnostic protocol (built, but deliberately not used for dataset ground truth)

Code: `btp-pipeline/src/diagnose.py`. A retry-then-reprobe heuristic that *guesses* the fault type purely from observed uncertainty patterns:

1. Same-input retry (k samples, identical faulted prompt) recovers → guess `noise`.
2. Clean-input retry (k samples, known-good input) recovers → guess `contamination`.
3. Neither recovers → guess `ceiling`.

This exists as a **non-learned comparison baseline**, and is deliberately never invoked during dataset generation — the dataset's ground truth comes from the known, deliberate fault injection itself, not from this heuristic's guess. `retries=2` (not 1) on the same-input check, since a single lucky retry is weak evidence against a "soft ceiling" that intermittently recovers by chance.

### 2.5 Local validation study (Study 1, 45 trials)

Before scaling up, the pipeline was validated locally (chain topology only, Apple Silicon / MLX backend, 45 trials — 15 each of noise/contamination/ceiling).

**Per-node uncertainty statistics:**

| Fault Type (n=15 each) | U_retriever mean (max) | U_reasoner mean (max) | U_writer mean (max) |
|---|---|---|---|
| Noise (temp=1.2) | 0.089 (0.333) | 0.667 (0.667) | 0.111 (0.333) |
| Contamination (distractors) | **0.222 (0.667)** | 0.667 (0.667) | **0.244 (0.667)** |
| Ceiling (stripped gold) | 0.089 (0.333) | 0.667 (0.667) | 0.089 (0.667) |

**Confusion matrix** (per-node threshold: U_retriever > 0.3 OR U_writer > 0.3):

```
diagnosed      noise  contamination  ceiling
true
noise              9              0        0
contamination      0             10        0
ceiling            0              0        5
```

- 24 / 45 trials crossed the threshold (were diagnosable); **100% precision on all 24** — zero cross-fault misclassifications.
- 21 / 45 trials fell below threshold — the model answered correctly from parametric memory despite the injected fault, a natural detection floor.

**Key findings from this phase:**
1. **A single global uncertainty threshold does not work.** Retriever/Writer produce short, low-variance text; Reasoner produces verbose CoT with high structural variance even when correct. Recommendation carried forward: per-node threshold vectors, not one global cutoff.
2. **Parametric memory is a confound** — fault injection has no effect whenever the model can answer correctly from its own weights regardless of context quality. This motivated the multi-topology, multi-fault-target design of Phase 2 (spreading faults across more nodes/structures to surface genuine failure modes a single-chain, retriever-only design would miss).
3. **Contamination produced the strongest, most consistent signal** in this local run (elevated both Retriever and Writer uncertainty) — this pattern re-emerged later at dataset scale (see §5).
4. **Topology structure changes fault *cost*, not just fault *signal*** — generation time is driven by the longest sequential dependency-chain depth and how many Reasoner-role nodes land in the same execution round (Reasoner uses a larger token budget), not raw node count.

---

## 3. Phase 2 — Dataset generation at scale (complete)

### 3.1 Backend and infrastructure

Generation was scaled from local Apple-Silicon/MLX prototyping to GPU-batched inference on free-tier cloud GPUs:

- **Model**: `Qwen/Qwen2.5-7B-Instruct-AWQ` (4-bit AWQ quantization, `float16` compute — T4/Turing GPUs lack native bf16). An MXFP4-quantized model was originally planned but doesn't run on T4 (needs Hopper/Ada-class hardware), which is what forced this substitution.
- **Serving**: vLLM, using `llm.chat(List[List[message]], List[SamplingParams])` for genuine cross-trial batching — all pending LLM calls across a topology's execution round are dispatched together, which is what makes hundreds of trials tractable on a free-tier GPU.
- **Hardware**: Kaggle Notebooks (free T4, ~30 GPU-hours/week quota) and, in parallel on disjoint question sets, Lightning AI (Studios + Jobs, free-tier T4) to roughly double effective throughput.
- Numerous concrete infrastructure practices (Kaggle Dataset+Kernel code delivery pattern, real-time log streaming to avoid mistaking a live run for a hang, NumPy 1.x/2.x ABI conflicts on Lightning AI, concurrent-GPU-machine limits, CPU-only kernels for non-inference work) are documented in `docs/infra/kaggle-and-lightning-setup.md`.

### 3.2 Topology pool

`dataset/topology_pool.json`, generated by `dataset/generate_topology_pool.py`: **20 topologies total**.

- **14 train topologies** (3–6 nodes): `chain`, `dual_retriever_fanin`, `triple_retriever_fanin`, `dual_retriever_fanin_deep`, `parallel_reasoner`, `triple_parallel_reasoner`, `crossed_fanin_fanout`, `deep_chain`, `deep_chain_5node`, `star`, `tree`, `wide_fanin`, `wide_fanout_reasoner`, `mixed_asymmetric`.
- **6 out-of-distribution (OOD) topologies** (`ood_random_0` through `ood_random_5`, 5–7 nodes): randomly generated via role-before-edges construction (valid by construction, no rejection sampling needed), structurally distinct from every train topology, **held out entirely** from training and hyperparameter tuning — used only for final generalization testing.

### 3.3 Fault grid and question source

- **Fault grid per topology**: 1 clean control + (3 fault types × N nodes). The 3 smallest topologies get the **full grid** (every fault type × every node); the remaining 17 get **partial coverage** (control + at least one fault type per node, cycling noise→contamination→ceiling by node position) — trading exhaustive per-topology coverage for broader structural diversity within a fixed compute budget.
- **Question source**: HotpotQA (distractor setting) validation split. Original run used 30 questions (15 seed + 15 sampled); later expanded to a disjoint 60+30-question pool for the larger dataset.

### 3.4 Real bugs found and fixed (via direct content inspection, not just "did it crash")

The single most important methodological lesson of this phase: a generation run completed with **zero crashes and correct-looking record counts** while still containing real correctness bugs — these only surfaced when someone actually opened generated records and checked whether the *labeled* fault visibly changed the model's output.

1. **Ceiling role-dispatch bug** (`src/faults.py`) — `inject_ceiling()` compared the node's *ID string* (`"retriever"`) instead of its *role*. Non-chain topologies use custom node IDs (`retriever_a`, `n0_r`, ...), so ceiling faults silently fell through to an "unsupported" skip on every topology except the plain 3-node chain. Fixed by threading an explicit `role` parameter through `inject_ceiling()` and dispatching on role everywhere.
2. **Hardened instruction never applied** (`src/pipeline.py`) — `build_prompt()` computed the hardened (ceiling-faulted) instruction but then used the original, unmodified instruction in the final prompt string regardless — every downstream-targeted ceiling trial ran with **zero actual effect**, despite being logged as `"ceiling"`. Fixed by routing the selected (hardened-when-applicable) instruction into the final prompt, verified with a 4-point unit test (normal case, faulted case, normal-absent-when-faulted, no cross-node leakage).
3. **Contamination self-substitution** (`src/faults.py`) — the distractor substitute bank could select the clean value itself as its own "wrong" substitute, producing a no-op fault. Fixed by excluding the clean parent output from the substitute bank before sampling (the trial is skipped, with a logged reason, if the bank becomes empty as a result).

All three fixed, and **the entire dataset was regenerated from scratch** afterward (not patched/merged in place), so the final dataset is internally consistent. Fixing bugs #1 and #2 together increased ceiling-fault representation from 20 → 103 records (5.15×) on regeneration, since ceiling faults now actually reached the model.

**Verification, not just a count check**: downstream (writer-targeted) ceiling trials post-fix show genuinely constrained output (e.g. `"American nationality."`, `"New York City"` — 2–5 words, matching the "at most 5 words" hardened instruction) instead of the old unconstrained full-sentence answers that had silently ignored the fault.

**Known, accepted, not-fixed limitation**: contamination on a multi-parent node (e.g. a `reasoner` fed by two retrievers) only corrupts the *first* parent's output — the other parent stays clean. This doesn't cause a wrong label, just a weaker-than-ideal signal for those specific trials. Flagged as a caveat for the thesis write-up, not planned to be fixed.

### 3.5 Final dataset

Two generations exist in the project's history:

- **Original, fully verified run**: 493 trial records (15 skipped, all genuine `ceiling_answer_survived`), 329 train-topology / 164 OOD-topology.
- **Expanded run** (current, used for all Phase 3 results below): **1,762 trial records**, generated with a larger disjoint question pool. Verified composition (recomputed directly from `dataset/trials.jsonl`, not from any summary doc):
  - By fault type: `clean=262`, `noise=578`, `contamination=550`, `ceiling=372`
  - By split: `train=1188`, `ood_test=574`
  - Topology sizes: 1×3-node, 3×4-node, 9×5-node, 6×6-node, 1×7-node (20 topologies total)
- `dataset/skipped.jsonl` = 15 records, all logged with an explicit reason, never silently dropped.
- **Generation parameters**: `k=5` self-consistency samples per node, default temperature 0.7, noise-fault temperature 1.2, `MAX_TOKENS=350` (Retriever/Writer), `MAX_TOKENS_REASONER=600`.

Full field-by-field record schema (including exactly which fields are model input, which are ground truth, and which are bookkeeping-only) is in `docs/dataset/dataset_description.md`.

**Known, accepted data-coverage gap**: in the `star` and `triple_retriever_fanin` topologies, the `retriever_c` node is never once used as a fault target in the dataset (root cause: the fault-assignment cycling logic happens to always land `retriever_c` on `ceiling`, which fails its survival-verification check for that specific node every time). Decision: not fixed/regenerated — this just means `retriever_c` is excluded from the candidate/target-node set when scoring those two topologies specifically; it was never a possible correct answer to begin with, so a model never predicting it there is not a failure.

### 3.6 Dataset-generation operational practices — every practice used, in full detail

Generation at this scale (hundreds to thousands of LLM-batched trials, on free-tier cloud GPUs, across two independent cloud platforms) required a specific set of operational practices, each one arrived at by actually hitting the problem it solves. These are documented in full in `docs/infra/kaggle-and-lightning-setup.md`; captured here in complete detail because they are as much a part of "what was done" as the modeling work itself.

#### 3.6.1 Choice of platform, and why both were used

- **Kaggle**: free NVIDIA T4 GPU, ~30 GPU-hours/week quota per account, kernel-based workflow (push code + a dataset, run a script end-to-end, pull logs/output afterward). This was the proven, primary platform — used for the original 493-record run and the first stage of the ~950/1762-record expansion.
- **Lightning AI**: free-tier T4 GPU via "Studios" + "Jobs" (scriptable through the `lightning_sdk` Python package). Used to run a **second, fully parallel** generation job on a disjoint question set, roughly doubling effective throughput without waiting on Kaggle's queue or weekly quota.
- Running both simultaneously on genuinely disjoint data (different questions, so no overlap/duplication risk) was treated as a legitimate scaling strategy, provided both sides are kept consistent (same code, same model, same bug-fix state — see §3.6.6).

#### 3.6.2 Kaggle workflow, step by step

- **Authentication**: `pip install kaggle` then `kaggle auth login --force` (opens a browser OAuth flow). Known gotcha: `kaggle auth print-access-token` can return a token that has gone stale after sitting idle for hours, producing an `Authentication required` error despite looking authenticated — the fix is simply re-running `auth login --force`.
- **PATH gotcha (Windows)**: the `kaggle` CLI script can land in `%APPDATA%\Roaming\Python\Python3xx\Scripts`, which is not always on PATH by default. If `python -m kaggle ...` works but bare `kaggle ...` does not, that directory needs to be added to PATH permanently.
- **Code delivery pattern — a Dataset + a Kernel, never a Notebook with inline code**: the pipeline code (`src/`, `scripts/`, `data/`, `dataset/topology_pool.json`) is packaged as a private **Kaggle Dataset** (Kaggle's generic file-storage primitive, unrelated to "ML datasets" in the usual sense), then referenced from a separate **Kernel** (script type, GPU + internet enabled). This decouples "iterate on code" (push a new Dataset version) from "manage a compute run" (launch/monitor/kill a Kernel) — far cleaner than embedding code directly in a Notebook cell.
  ```
  cd <staged_package_dir>
  kaggle datasets version -p . --dir-mode zip -m "<message>"
  ```
- **Critical gotcha — verify every push actually applied**: `kaggle datasets version` can print "Upload successful" for every file **without the new content actually going live** (observed at least once in practice). The fix is to always verify afterward, every single time, not just when something looks wrong:
  ```
  kaggle datasets files <owner>/<dataset-slug>
  ```
  checking that file sizes/timestamps match what was just pushed, especially for any file whose byte size changed.
- **Kernel mount path is not what Kaggle's own docs examples suggest**: a Dataset attached to a Kernel mounts at `/kaggle/input/datasets/<owner>/<dataset-slug>`, **not** the shorter `/kaggle/input/<dataset-slug>`. Verified with a quick `os.listdir()` debug print at the top of the kernel entry script before assuming paths are broken elsewhere.
- **`scripts/` is not a Python package on Kaggle**: `CODE_ROOT/scripts` must be added to `sys.path` and script modules imported as top-level (`from run_study2 import X`), not `from scripts.run_study2 import X` — there is no enforced `scripts/__init__.py` convention identical to a local checkout.
- **Real-time log streaming is mandatory, not optional, for any multi-hour run**:
  ```python
  proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
  for line in proc.stdout:
      print(f"[{time.strftime('%H:%M:%S')}] {line.rstrip()}", flush=True)
  proc.wait()
  ```
  `subprocess.run(capture_output=True)` must **not** be used for the main generation call — it buffers all child output until the process fully exits, making a multi-hour run indistinguishable from a genuinely hung one. This exact mistake cost roughly an hour of confused waiting on an early run before being diagnosed.
- **Watching a running kernel's logs**:
  ```
  export PYTHONUTF8=1   # avoids a Windows console-encoding crash on em-dashes/Unicode in log content
  timeout 130 kaggle kernels logs -f <owner>/<kernel-slug> > /tmp/log.txt 2>&1
  grep -E "=== |records written" /tmp/log.txt
  ```
  `kernels logs -f` **always replays from the start** of the log, so a short `timeout` window can miss recent lines on a long-running kernel — a 130-second-or-longer budget is needed to reach recent output, or the check simply needs to be re-run periodically.
- **Checking kernel status**: `kaggle kernels status <owner>/<kernel-slug>`. There is no clean "pause" or "stop" command — only `kernels delete`, which removes the kernel object entirely (destructive, but re-pushing recreates it). Practice adopted: always use a fresh, distinctly-named kernel ID/title per launch, rather than reusing an old name, to avoid confusing a stale cached kernel page with a genuinely new run.
- **`--resume` at topology-level granularity, for any run estimated at multiple hours**:
  ```python
  parser.add_argument("--resume", action="store_true",
      help="Skip topologies already complete in --log-path (topology-level granularity)")
  ```
  A killed/restarted run then redoes at most the one topology that was in progress when it died, not the entire run — cheap insurance against Kaggle's session limits or an unexpected crash.
- **Real timing data used for planning/ETA estimation**: per-topology generation time correlates with **how many Reasoner-role nodes execute in the same batched round**, not simple node count or chain depth (the Reasoner uses a much larger token budget for its chain-of-thought):

  | Reasoner nodes in same round | Typical duration |
  |---|---|
  | 1 | ~7–9 min |
  | 2 | ~9–11 min |
  | 3 (sequential) | ~12 min |
  | 3 (parallel, same round) | ~22 min (worst case) |
  | 4 (parallel) | ~21 min |

  Model load itself is a one-time ~8–10 minute cost at kernel start, not a per-topology cost.

#### 3.6.3 Lightning AI workflow, step by step

- **Setup**: `pip install lightning-sdk`, an API key from lightning.ai (Settings → Keys), and the account's **username string** (not the numeric User ID shown elsewhere on the same page — the SDK needs the username, not the UUID). Credentials stored in a gitignored JSON file (e.g. `lightning_credentials.json`), never committed — same handling as any other credential.
- **Resolving the correct teamspace**: `User().teamspaces` can show a teamspace name (e.g. `'general'`) that then 403s when used directly, because a personal account's actual GPU-billing teamspace is often **owned by an auto-created organization**, not the user directly. Lightning's own error messages are informative here (e.g. naming which organization the account is actually a member of) and should be read carefully rather than guessed around; if a plain `teamspace=` guess fails, listing `Organization(<org-name>).teamspaces` reveals the real internal name (commonly `default-project`, not the display name shown in the UI).
- **Creating a Studio and uploading code**:
  ```python
  studio = Studio(name="btp-dataset-gen", teamspace="default-project", org="yourname-org", create_ok=True)
  ```
- **Critical Windows-specific bug — never use `Studio.upload_folder` from a Windows client**: its internal `upload_file` call applies `os.path.normpath(remote_path)`, which on Windows converts forward slashes to backslashes — so a remote path like `btp_code/data/foo.json` gets silently mangled into a **flat file literally named** `btp_code\data\foo.json` (a literal backslash character in the filename, not a directory separator) on the Linux remote. The resulting Job then fails with `cd: btp_code: No such file or directory`, even though a file listing appears to show files "under" that path. The workaround adopted: bypass the buggy high-level wrapper and call the lower-level upload API directly with POSIX-style paths, uploading file by file:
  ```python
  for dirpath, dirnames, filenames in os.walk(local_root):
      for fn in filenames:
          rel = os.path.relpath(os.path.join(dirpath, fn), local_root).replace(os.sep, "/")
          remote_path = "btp_code/" + rel
          studio._studio_api.upload_file(..., remote_path=remote_path, ...)
  ```
  then verifying with `studio.run("find btp_code -type f | sort")` (which requires the Studio to be in the `Running` state — `studio.start()` first if `Stopped`) to confirm genuinely nested paths rather than flat, backslash-mangled filenames.
- **Also avoid empty (0-byte) files in the uploaded tree** — e.g. a bare `src/__init__.py` with no content was observed causing an HTTP 501 mid-upload with the buggy `upload_folder` path; every `__init__.py` was given at least one line (a package-marker comment) to avoid this.
- **Launching a generation Job**:
  ```python
  cmd = (
      "cd btp_code && pip install -q -r requirements.txt && pip install -q vllm && "
      "pip install -q --upgrade --force-reinstall --no-cache-dir scipy numba mistral-common matplotlib && "
      "mkdir -p logs && "
      "BTP_BACKEND=vllm python -u scripts/run_study_vllm.py "
      "--topology-pool dataset/topology_pool.json --examples-json data/my_questions.json "
      "--k 5 --batch-size 15 --examples-per-topology-full-grid 4 --examples-per-topology-partial 7 "
      "--log-path logs/dataset_trials.jsonl --skip-log-path logs/dataset_skipped.jsonl --resume"
  )
  job = studio.run_job(name="btp-dataset-gen-run1", machine=Machine.T4, command=cmd)
  ```
- **The NumPy 1.x/2.x ABI conflict — expected on every fresh Lightning environment, not a one-off bug**: Lightning's base "cloudspace" conda environment ships several packages pre-compiled against NumPy 1.x (`scipy`, `matplotlib`, `numba`, and transitively `torchmetrics`/`lightning.pytorch`, which the environment's own `sitecustomize.py` auto-imports on **every single Python invocation**, including scripts that never explicitly import them). Installing `vllm` silently upgrades NumPy to 2.x as a dependency without upgrading those other packages, so the very first `import scipy.signal` (or anything importing `torchmetrics`) crashes with an ABI-incompatibility error. Critically, **this is not a one-shot fix — fixing one package uncovers the next broken one in the same import chain** (`torchmetrics → scipy.signal` first, then `→ matplotlib`, then a `numba` version-pin conflict with `vllm`'s own pin). The sequence that actually worked, arrived at by iterating through the chain:
  ```
  pip install -q -r requirements.txt
  pip install -q vllm
  pip install -q --upgrade --force-reinstall --no-cache-dir scipy numba mistral-common matplotlib
  ```
  Two details that made earlier attempts fail silently: (1) `--force-reinstall` **alone, without `--upgrade`**, just reinstalls whatever version pip had already resolved — it does not guarantee a NumPy-2-compatible version, both flags are needed together; (2) `pip check 2>&1 | grep numpy` surfaces exactly which packages still declare a `numpy<2` requirement, rather than discovering them one crash at a time. A `vllm==X` pin on an exact `numba` version can conflict with the bumped numba — in practice this conflict was **cosmetic** (a pip warning, not a runtime failure), verified by testing the actual import chain end-to-end rather than trusting `pip check`'s warning alone.
  **Debug workflow adopted** (much faster than relaunching a full Job each time, which costs ~5–10 minutes of machine-setup overhead per attempt): install and test directly on the Studio's own persistent shell (separate from any Job's isolated snapshot) via `studio.run(...)`, and verify the actual import chain with an uploaded test script rather than an inline `python -c "..."` through `studio.run()` (multi-layer shell quoting through Python→shell→remote-python reliably mangles nested quotes/escapes). Only once a plain `test_imports.py` (`import scipy.signal; import matplotlib; import numba; from vllm import SamplingParams; import src.faults`) passes cleanly on the Studio shell is the full Job relaunched.
- **Concurrent GPU capacity limit**: a Job whose internal script crashes does **not** automatically stop the underlying machine — `job.status` can stay `Running` indefinitely even after the process inside has already exited with an error. Relaunching without first stopping the old one can leave the new Job stuck in `Pending` forever, silently blocked by the account's concurrent-GPU-machine limit (observed: 2 concurrent T4 machines on a free-tier account), with the tell-tale symptom being `job.started_at is None` after many minutes. Practice adopted: always explicitly `Job(...).stop()` dead/crashed jobs before relaunching, and check `job.status` across *all* recent launch attempts, not just the latest one, when a new launch seems stuck.
- **Retrieving generated output mid-run or after completion**: a Job's working directory is an isolated snapshot taken at launch time — **not** live-shared with the Studio's persistent storage, so files a Job creates while running (e.g. `logs/dataset_trials.jsonl`) are invisible via `studio.run("ls ...")`. The Job's own artifact API must be used instead: `job.list_artifacts(path=...)` to confirm a file exists (with a real, growing size) before `job.download_artifacts(target_dir=..., path=...)`. Known quirk: `list_artifacts` can show a file with a real, growing size while `download_artifacts` simultaneously reports "No files found" for the same path — an eventual-consistency lag in the artifact indexer for a file a still-running Job is actively writing to. The correct response is to wait for the Job to progress further (or fully complete) and retry, not to assume the file is genuinely missing.
- **Monitoring a running Job**: `job.status` transitions `'Pending' → 'Running' → 'Completed'/'Failed'/'Stopped'`; `job.logs()` raises `RuntimeError` while status is `'Pending'` (must be wrapped in try/except); `job.machine` confirms which GPU actually got allocated (e.g. `'T4'`). Polling should use a fixed sleep interval, and `Pending` should not be assumed stuck within the first few minutes — machine provisioning plus Studio snapshotting routinely takes 1–3 minutes before a Job transitions to `Running`.

#### 3.6.4 Shared practices applied to both platforms

- **Verify a data-quality claim by inspecting actual content, not just "did it run"** — the single most valuable practice from this project's entire history (this is exactly the discipline that caught the three real bugs in §3.4). A full generation run can complete with zero crashes and correct-looking record counts while still containing real correctness bugs. The only way to catch this is to manually spot-check a handful of records per fault type and actually read the `samples` field, before trusting a run's output, on either platform.
- **Back up before regenerating** — before replacing a dataset file with a new run's output, the old one is copied aside (e.g. `trials_v2_493.jsonl`) rather than overwritten in place. Regeneration runs cost hours of GPU time; a bug discovered in a new run should never cost the old, already-working dataset too.
- **Keep parallel runs on genuinely disjoint data** — when running Kaggle and Lightning AI (or any two compute sources) in parallel to increase total volume, each must draw from a **disjoint** question set, verified explicitly in code (an assertion that the two question sets' intersection is empty) *before* launching, not discovered after. Otherwise the two outputs end up with duplicated (question, topology, fault) recipes, complicating any later train/val split and wasting compute re-deriving information already available.
- **Same code, same model, on every parallel run** — any bug fix or config change made on one platform must be mirrored on the other before both sides are considered part of the same "dataset version," or the combined output ends up internally inconsistent (part of it carrying a bug the other part has already had fixed).
- **Don't request a GPU for CPU-bound work** — a Kaggle kernel that only needs CPU (feature enrichment via sentence-transformers, small-GNN training, hyperparameter sweeps) but is launched with `"enable_gpu": true` competes for the same limited GPU-slot queue as the actual data-generation kernels. Symptom actually observed: a CPU-bound kernel stuck `QUEUED` with a completely empty log for 15+ minutes across four relaunch attempts (two stuck, two cancelled) — looked like a code bug, was not; a minimal `enable_gpu: false` test kernel (no dataset, just a print statement) scheduled and completed instantly on the same account at the same time, confirming the bottleneck was GPU-queue contention. Fix adopted: `"enable_gpu": false` in `kernel-metadata.json` for anything that doesn't do LLM inference — `torch`/`torch_geometric` training on graphs this small runs fine on CPU by design, this practice just needed to be reflected in the actual kernel config once the symptom forced the question.

#### 3.6.5 Scaling generation across more than two compute sources

When one Kaggle account plus one Lightning account was not enough parallelism (e.g. for the higher-cost k=10 regeneration, which costs roughly proportionally more GPU time per topology than k=5), the same patterns above were designed to compose across additional accounts:

- **A third Kaggle account**: identical setup to §3.6.2, with a new account's API token exported as an environment variable for that specific command rather than overwriting the default credentials file (which stays bound to the main/first account) — verified by a cheap authenticated call (`kaggle datasets list --mine`) before assuming the export actually took effect.
- **A second Lightning AI account**: same Studio/Job pattern as §3.6.3, but the teamspace ownership model must be re-checked on the new account, since it can differ from the first account's (e.g. the first account's teamspace was org-owned; a second account's personal teamspace may be owned directly by the `User`, not an `Organization`). The practice adopted: try the plain `teamspace='default-project', user='<username>'` form first, only falling back to the `org=` parameter if that 403s.
- **Splitting the work**: each compute source is given a disjoint slice — a disjoint question set, or a disjoint topology range — so outputs merge cleanly afterward with no duplicate-recipe risk, the same principle as §3.6.4 extended to N sources instead of 2.
- **Handling one source running out mid-generation gracefully**: if a source (typically the one with the tightest free-tier budget) exhausts its quota/credits partway through, whatever partial output it produced is pulled down (the same per-file artifact-download pattern as §3.6.3's Windows-upload workaround, in reverse), and the *remaining* topologies are resumed on a source that still has capacity — seeding that source's `logs/` directory with the partial output before invoking `--resume`, exactly the same recovery pattern used for a single-source crash, just re-targeted at a different compute source than the one that originally produced the partial data. This is the exact pattern used for the k=10 preview data currently sitting at 1,278/expected-full records (§7.2).

#### 3.6.6 Why these practices matter for the thesis write-up specifically

None of this is incidental tooling trivia — every one of these practices exists because it was the direct fix for a real failure mode actually hit while generating this project's data (a stale auth token, a false-positive "upload successful," a wrong mount path, a silently-hanging log, a mangled Windows upload path, a NumPy ABI chain, a GPU-queue deadlock, a partial-output recovery). Collectively, they are the operational backbone that made it possible to generate a multi-thousand-trial, multi-topology, bug-verified dataset on exclusively free-tier compute, and are worth summarizing in the thesis's methodology or appendix as evidence of engineering rigor, not just narrated as background color.

---

## 4. Phase 3 — GNN fault-localization model (complete for the k=5 dataset)

All model code lives in `./model/` (a hard project rule — never inside `btp-pipeline/src/`, which is scoped to the dataset-generation pipeline only).

### 4.1 Data pipeline

`trials.jsonl` + `topology_pool.json` → per-node feature vectors → PyTorch Geometric `Data` graph objects → split → baselines → GNN.

- **Node feature vector** evolved over the course of the project (see §4.3): started at 8 scalar dimensions, ended at 268 dimensions (scalars + PCA-compressed embedding mean + PCA-compressed embedding variance).
- **Missing-value handling**: metrics that don't apply to a given role (e.g. `semantic_uncertainty` for a retriever) get an explicit **mask bit** (`has_semantic`, `has_jaccard`) alongside a placeholder `0`, rather than a silent, ambiguous `0` — the model can tell "measured as zero" apart from "not applicable."
- **Node label**: one-hot over the graph's own nodes, `1` at `fault_config["target_node"]`'s index; clean/control trials get an explicit extra **"no fault" class** (not an all-zero vector, which is ill-defined under softmax) — a `k+1`-way softmax per graph (its N real nodes, plus one "no fault").
- **Edges**: directed from `topology_pool.json`, with a bidirectional variant (forward + reverse) tested and adopted (see §4.3) — the reasoning being that a fault's *symptom* often needs to propagate backward, via message passing, from a downstream node toward its true upstream cause.
- **Splitting**:
  - **OOD test set** = all trials whose topology's `split == "ood_test"` (574 trials, 6 topologies) — never touched during training or hyperparameter tuning, at any point.
  - **Train/validation split** within the 14 train topologies (1188 trials) is grouped **by question, not by trial** — the same 30–90 source questions recur across many topologies and fault conditions, so a naive per-trial random split would leak a question's exact phrasing between train and validation.

### 4.2 Baselines (built and evaluated first, to have a number the GNN must beat)

| Approach | OOD top-1 accuracy |
|---|---|
| **Naive max-uncertainty** — pick whichever node has the highest lexical uncertainty, no learning at all | **0.145** |
| **Non-graph RandomForest** — same per-node scalar+embedding features, flattened to a fixed-size table, standard RandomForest classifier (tests whether graph *structure* is adding anything, independent of model family) | **0.249** |

### 4.3 GNN development — the full ablation history

Model: `model/gnn.py`'s `FaultLocalizerGNN` — a PyTorch Geometric message-passing network (configurable GAT / GCN / SAGE convolution), with an input-projection + dropout stage before the first graph layer, a node-level classification head, an optional graph-level "no fault" head (pooled from node embeddings), and an optional multi-task fault-type auxiliary head. Every change below is logged, individually, in `model/metrics_history.jsonl` (43 runs) — nothing here is an unverifiable aggregate claim.

| # | Change | Dataset | OOD top-1 | Note |
|---|---|---|---|---|
| 1 | First GNN (GAT, 8-dim scalar features, forward-only edges) | 493 records | 0.220 | Baseline GNN before dataset expansion |
| 2 | Dataset expanded | 1762 records | 0.206 | Larger dataset, same architecture — a real short-term dip, not yet a regression once later changes are added |
| 3 | + Bidirectional edges | 1762 | 0.232 | Small, consistent, cheap win — kept permanently from here on |
| 4 | + Raw 384-dim sentence embeddings, no regularization | 1762 | 0.239 | Looked much better on train/val (top1 jumped 0.30→0.54) but **barely moved OOD** — classic overfitting from feeding a large raw feature space into a tiny GNN. A trap if only train/val numbers are checked. |
| 5 | + Input projection layer + dropout(0.3) + PCA-32 compression of the embeddings (fixes #4's overfitting) | 1762 | 0.263 | This is what actually made embeddings pay off on OOD — not the embeddings themselves, but the regularization that let the model use them without memorizing |
| 6 | + Hyperparameter sweep winner (hidden_dim=64, num_layers=2, lr=1e-3) | 1762 | 0.275 | num_layers=3 uniformly *hurt* at this point (over-smoothing on 3–7 node graphs); lr=5e-4 was uniformly worse than 1e-3 here |
| 7 | Multi-task fault-type auxiliary head tried | 1762 | 0.267 | Neutral-to-slightly-negative; not adopted as-is |
| 8 | Architecture sweep: GAT vs GCN vs SAGE, matched hyperparameters | 1762 | GAT 0.275 / GCN 0.246 / **SAGE 0.359** | SAGE's fixed/mean neighbor aggregation clearly outperforms GAT's learned attention and GCN's fixed spectral aggregation on these small graphs |
| 9 | + `node_embedding_std` (embedding variance across the k samples) as an added feature, PCA-128 (mean+std fit independently) | 1762 | 0.378 | Confirmed a real, independent win on top of the SAGE switch |
| 10 | Pooling strategy sweep (mean / max / attention) for the graph-level head | 1762 | mean 0.375 / max 0.361 / **attention 0.387** | Attention pooling — a learned per-node weighting rather than uniform averaging — became the single best individual technique |
| 11 | 3-seed ensemble (average softmax across 3 independently trained SAGE models), attention pooling alone (not combined) | 1762 | 0.383 | A real but smaller win than attention pooling alone at that point |
| 12 | **Attention pooling + 3-seed ensemble stacked together** (the two independent wins, combined) | 1762 | **0.395** | **Current best result on the k=5 dataset.** OOD top-2 = 0.587, OOD precision = 0.427, OOD recall = 0.365, OOD F1 = 0.380. |

Also tried, with modest/mixed results not adopted as the final configuration: label-smoothing and focal loss (small, inconsistent gains), PCA dimensionality sweep (128 components outperformed 32/64 once the variance feature was added), a non-graph RandomForest sanity check (0.249, confirmed above graph structure is adding real value beyond it).

**Final comparison table** (OOD test set, n=574, all evaluated identically):

| Approach | OOD top-1 |
|---|---|
| Naive max-uncertainty baseline (no learning) | 0.145 |
| Non-graph RandomForest (flat features) | 0.249 |
| GNN — GAT | 0.275 |
| GNN — GCN | 0.246 |
| GNN — SAGE (base config) | 0.359 |
| GNN — SAGE + embedding variance feature | 0.378 |
| GNN — SAGE + attention pooling | 0.387 |
| GNN — SAGE + 3-seed ensemble (no attention pooling) | 0.383 |
| **GNN — SAGE + attention pooling + 3-seed ensemble (final, best)** | **0.395** |

### 4.4 Full metric breakdown of the final model

Final config: GraphSAGE, 3 layers, hidden_dim=64, dropout=0.3, lr=5e-4, bidirectional edges, PCA-128 embeddings (mean + variance), attention pooling, 3-seed ensemble.

| Split | Top-1 Acc | Top-2 Acc | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Train | 0.69 | 0.85 | 0.68 | 0.70 | 0.67 |
| Validation | 0.42 | 0.60 | 0.42 | 0.40 | 0.40 |
| **OOD (held-out)** | **0.40** | **0.59** | **0.43** | **0.36** | **0.38** |

The train/val/OOD gap (0.69 → 0.42 → 0.40) shows real, expected overfitting to training-topology specifics, but validation and OOD tracking each other closely (0.42 vs. 0.40) indicates the val split is a reasonably honest proxy for OOD generalization at this point — not a false sense of security from a val set that's secretly leaking topology-specific shortcuts.

---

## 5. Deeper evaluation — ROC/PR curves, role breakdown, fault-type breakdown

These required actually re-running the final model locally (the aggregate metrics above come from `metrics_history.jsonl`, which stores summary numbers only, not raw per-node predictions) to get real, non-estimated scores. `torch_geometric` was installed locally for this; the reproduction run achieved OOD top-1 = 0.383 (consistent with the logged 0.395 — small difference is normal seed/CPU run-to-run variance).

### 5.1 Node-level fault identification as a pooled binary task

Treating every node instance across every OOD graph as one binary classification example ("is this the true fault node," n=3,962 node instances, ~14.5% positive rate):

- **ROC-AUC = 0.739** (0.5 = random).
- **Average Precision = 0.415** (vs. a 0.145 random/base-rate baseline) — the more informative of the two here, since positives are a minority class.

### 5.2 ROC by node role

| Role | AUC | n |
|---|---|---|
| Reasoner | **0.785** | 1946 |
| Writer | 0.732 | 770 |
| Retriever | 0.695 | 672 |
| "No fault" (clean graphs) | 0.675 | 574 |

Reasoner faults are the easiest for the model to identify; retriever faults and clean/no-fault graphs are the hardest.

### 5.3 Role-level confusion matrix (true role vs. predicted role, OOD set)

Row-normalized:

|  | Pred: retriever | Pred: reasoner | Pred: writer | Pred: no_fault |
|---|---|---|---|---|
| **True: retriever** | 28% | 41% | 9% | 22% |
| **True: reasoner** | 8% | **73%** | 8% | 11% |
| **True: writer** | 12% | 37% | 37% | 15% |
| **True: no_fault** | 15% | **50%** | 6% | 29% |

**Important finding, not to be glossed over**: the model over-predicts "reasoner" across the board — including 50% of genuinely clean/no-fault graphs getting misclassified as a reasoner fault. This means reasoner is partly acting as the model's majority-class fallback, not purely reflecting a genuinely stronger reasoner-specific signal. Worth stating explicitly as a limitation in the thesis Results/Discussion section, not hidden behind the strong reasoner AUC number in §5.2 alone.

### 5.4 OOD accuracy by true fault type

| Fault type | Top-1 accuracy | n |
|---|---|---|
| **Contamination** | **0.589** | 168 |
| Ceiling | 0.35 | 140 |
| Clean | 0.286 | 84 |
| Noise | 0.264 | 182 |

Contamination is clearly the easiest fault type to localize — directly consistent with the Phase 1 local-study finding (§2.5) that contamination produced the strongest uncertainty signal. Noise is the hardest, consistent with noise being the most "recoverable"/transient of the three fault types and therefore leaving the weakest structural fingerprint.

---

## 6. Summary of the k=5 era (Phases 1–3)

1. The core premise holds: per-node self-consistency uncertainty, combined with topology, is a genuine, learnable fault-localization signal — well above chance (ROC-AUC 0.739) and clearly above both a naive per-node rule (0.145) and a non-graph model on the same features (0.249).
2. Graph structure specifically matters, not just richer features: the GNN family (0.246–0.395 depending on configuration) consistently outperforms the non-graph RandomForest (0.249) once architecture and regularization are tuned, and SAGE-style neighbor aggregation in particular outperforms GAT/GCN on these small (3–7 node) graphs.
3. A single global uncertainty threshold is provably insufficient (Phase 1), and even a well-tuned GNN still shows real, uneven weaknesses by fault type (noise hardest), by node role (retriever hardest, reasoner over-predicted as a fallback), and by split (real but moderate overfitting from train to OOD).
4. Data-quality discipline — verifying actual generated content, not just structural/count checks — caught three real, silent correctness bugs in the fault-injection pipeline that would otherwise have corrupted the ground truth the entire modeling phase depends on.
5. The finished-dataset headline number is **OOD top-1 accuracy = 0.395** (macro-F1 ≈ 0.38) on the k=5, 1762-record dataset, reached through a documented sequence of individually-verified architecture and feature changes, evaluated throughout with strict separation between in-distribution and out-of-distribution performance.

---

## 7. Phase 4 — k=10 dataset regeneration (in progress, not yet final)

### 7.1 Motivation

At k=5, lexical uncertainty can only take 6 discrete values (`{0, 0.2, 0.4, 0.6, 0.8, 1.0}`), which coarsens the uncertainty signal the whole approach depends on. Regenerating with **k=10 self-consistency samples per node** gives 11 discrete levels instead, a finer-grained signal, at roughly double the GPU cost per topology. This was queued as the next concrete lever to close the gap toward a higher target accuracy, run in parallel across multiple Kaggle accounts and a Lightning AI account (same multi-source parallelization pattern as the original k=5 expansion — see `docs/infra/kaggle-and-lightning-setup.md` §7).

### 7.2 Status as of this writing

**Not finished.** A partial file exists locally at `dataset/trials_k10_preview.jsonl` — **1,278 records**, nominally covering all 20 topologies but with 4 train topologies still under-populated (per the run's own logged note: "16/20 topologies, 4 train topologies still missing" — meaning functionally incomplete coverage for those 4, not literally zero records). An earlier, smaller preview (1,099 records) also exists in the run history but not as a standalone file locally.

A dedicated script, `model/preview_run.py`, exists specifically to validate the pipeline early against this partial data and get an early accuracy signal — explicitly documented as **not the final reported number**.

### 7.3 Preview results so far

Both runs used the exact winning k=5 architecture (GraphSAGE, hidden_dim=64, num_layers=3, dropout=0.3, lr=5e-4, bidirectional edges, PCA-128 mean+variance embeddings, attention pooling, 3-seed ensemble) against the partial k=10 data, logged in `model/metrics_history.jsonl`:

| Preview run | Dataset size | OOD top-1 | OOD n |
|---|---|---|---|
| Preview #1 | 1,099 records (16/20 topologies) | 0.417 | 345 |
| Preview #2 | 1,278 records (20/20 topologies, 4 train topologies under-populated) | 0.409 | 345 |

Both numbers are **higher** than the finished k=5 dataset's 0.395, which is an encouraging early signal for the finer-grained k=10 approach — but both are explicitly logged as previews on an admittedly incomplete dataset, and the OOD set size itself (n=345) differs from the finished dataset's n=574, so these numbers are **not directly comparable** to the k=5 headline result yet. Treat as "promising, not yet load-bearing."

### 7.4 Attempted deeper evaluation (ROC/PR/confusion) on the k=10 preview — incomplete

An attempt was made to reproduce the k=10-preview run locally (mirroring `model/preview_run.py`, including the sentence-embedding enrichment step for the 1,278 un-enriched preview records) in order to get the same kind of ROC/PR/role-confusion/fault-type breakdown already produced for the k=5 model (§5). This run (`make_roc_charts_k10.py` in the session scratchpad) was **killed partway through by the system's background-process memory reaper** (the host ran low on memory while the session was idle — not a bug in the script or the model code) and was not restarted. **No ROC/PR/confusion-matrix numbers exist yet for the k=10 preview data.** This is an explicit open item (§9).

---

## 8. Report-writing and documentation produced

Alongside the code/data work, a full first-draft set of BTech thesis (BTP) report sections has been written, based on the real project state at the time each was drafted:

1. **Introduction** — the multi-agent silent-failure framing, the core hypothesis, and the two/three-phase project structure.
2. **Motivation** — why output-level confidence checks are insufficient for multi-agent systems, the localization + fault-type-differentiation argument, and the research-novelty argument (intersection of LLM uncertainty literature and multi-agent-systems literature).
3. **Problem Statement** — the formal node-classification-over-a-graph framing, and the two non-trivial requirements (correct attribution despite downstream symptom displacement; generalization to unseen topologies).
4. **Simulation Platform and Requirements** — hardware (Apple Silicon dev, Kaggle/Lightning AI T4 GPUs for generation, local CPU for modeling), software stack, data requirements, and the specific infrastructure constraints hit and worked around.
5. **Conclusion** — written explicitly as **provisional/interim**, honestly stating that the project demonstrates the framework and methodology work, not yet that fault localization is "solved," with a note to revise once k=10 results are final.
6. **Objectives** — the seven concrete objectives spanning pipeline-building through generalization evaluation and error analysis.
7. **Methodology** — the full technical methodology write-up (pipeline/fault-injection design, dataset generation procedure, feature engineering, modeling/evaluation protocol), cross-referenced to the chart set in §9.

None of these sections have yet been consolidated into a single report file — they were drafted progressively in conversation and are recorded in full in the session history. **Not yet written**: Related Work / Literature Review (blocked on the user supplying the research papers the project was originally based on), a proper Results section text (the numbers and charts exist — §5, §6, §9 — but a prose Results section has not been written), Discussion/Limitations as a dedicated section, and References.

A separate handoff document, `docs/handoff/project-context-2026-09-27.md`, was also written specifically to seed a *new* chat session with full context — it duplicates much of what's in this file in a more conversational, "here's where we left off" format, and its own contents were updated at least once (best-result number corrected from 0.387 to 0.395) as new results landed. It has not been updated with the k=10 preview information in §7 above.

---

## 9. Charts and figures generated

All charts were generated from real project data (dataset files, `model/metrics_history.jsonl`, or a locally-reproduced model run) — never fabricated or estimated — and are saved permanently in `docs/results/figures/` (originally produced in a session scratchpad, then copied into the repo).

| File | Content | Source |
|---|---|---|
| `fig1_fault_type_distribution.png` | Dataset composition by fault type (clean/noise/contamination/ceiling counts) | `dataset/trials.jsonl` |
| `fig2_train_ood_split.png` | Train vs. OOD trial counts | `dataset/trials.jsonl` + `topology_pool.json` |
| `fig3_topology_size_distribution.png` | Topology graph-size distribution (node counts across the 20 topologies) | `dataset/topology_pool.json` |
| `fig4_ood_accuracy_progression.png` | OOD top-1 accuracy across every logged experiment run on the 1762-record dataset | `model/metrics_history.jsonl` |
| `fig5_architecture_comparison.png` | GAT vs. GCN vs. SAGE at matched hyperparameters | `model/metrics_history.jsonl` |
| `fig6_model_comparison.png` | Naive baseline vs. RandomForest vs. best k=5 GNN vs. k=10-preview GNN | `model/metrics_history.jsonl` |
| `fig7_best_model_metrics.png` | Best k=5 model's top-1/top-2/precision/recall/F1 across train/val/OOD | `model/metrics_history.jsonl` |
| `fig8_accuracy_evolution.png` | Milestone-by-milestone OOD accuracy from the first GNN run (0.220) through the k=5 best (0.395) to the two k=10 preview points (0.417, 0.409) | `model/metrics_history.jsonl` |
| `fig9_roc_curve.png` | Pooled binary ROC ("is this node the true fault source," n=3,962 node instances, k=5 model) — AUC 0.739 | Locally-reproduced model run |
| `fig10_pr_curve.png` | Same pooled binary task, Precision-Recall curve — AP 0.415 | Locally-reproduced model run |
| `fig11_roc_by_role.png` | ROC curve broken out by node role (reasoner/writer/retriever/no-fault) | Locally-reproduced model run |
| `fig12_role_confusion_matrix.png` | Row-normalized true-role vs. predicted-role confusion matrix, OOD set | Locally-reproduced model run |
| `fig13_accuracy_by_fault_type.png` | OOD top-1 accuracy split by true fault type (contamination easiest, noise hardest) | Locally-reproduced model run |

Generation scripts (also copied into `docs/results/figures/` for reproducibility): `make_report_charts.py` (figs 1–8) and `make_roc_charts.py` + `make_roc_plots.py` (figs 9–13, requires re-training the k=5 ensemble locally to dump real per-node prediction probabilities, since `metrics_history.jsonl` only stores aggregate metrics, not raw scores).

**Not yet produced**: the equivalent ROC/PR/role-confusion/fault-type-breakdown charts for the k=10 preview data — attempted, but the training run was killed by a system memory reaper mid-run (§7.4) and has not been re-run.

---

## 10. Tooling and infrastructure notes from the modeling/evaluation session work

- **`torch_geometric` was not originally installed in the local Python environment** used for chat-session work (only `torch` 2.7.1+cpu was present) — installed via `pip install torch_geometric` to enable locally reproducing trained models for deeper evaluation. `sentence-transformers` and `scikit-learn` were already present.
- **Local CPU reproduction is genuinely feasible** for this project's model scale: reproducing the full 3-seed SAGE + attention-pooling ensemble (150 epochs requested per seed, early-stopped in practice at 37–91 epochs) on the k=5 dataset took only a few minutes on CPU, consistent with the project's own documented expectation that this phase needs no GPU.
- **The k=10 preview reproduction is heavier**: it additionally requires computing sentence embeddings from scratch for 1,278 un-enriched records (the k=5 dataset's `trials.jsonl` already has pre-computed `node_embeddings`/`node_embedding_std` fields; `trials_k10_preview.jsonl` does not, so `model/data/enrich_features.py`'s `enrich()` must run first as part of any reproduction). This is very likely why this run took long enough to still be running when the host's background-process memory reaper killed it.
- **A note on that memory-reaper kill**: it is host-level protective behavior (Claude Code stops background shell commands when the system is critically low on memory while a session is idle) and is not a verdict on the script's own memory usage or correctness. Per its own guidance, it should not be restarted automatically — only on explicit request.

---

## 11. Known open items, gaps, and next steps

**Data / generation:**
- Finish the k=10 dataset regeneration (4 train topologies still under-populated as of the last local preview file) and merge it into a final, verified `trials.jsonl` the same way the original 493→1762 expansion was verified (direct content inspection, not just record counts — this project has a demonstrated history of exactly this kind of bug hiding behind clean-looking structural checks, see §3.4).
- Re-run `scripts/verify_results.py`-style validation on the final k=10 dataset before trusting it for training, per the project's own established discipline.
- The `retriever_c` coverage gap (§3.5) and the multi-parent contamination partial-corruption limitation (§3.4) both carry over unchanged into the k=10 data, since they're structural/logic issues in the generation code, not k-specific.

**Modeling / evaluation:**
- Re-run the ROC/PR/role-confusion/fault-type-breakdown analysis (§5, §9) once the k=10 dataset is finalized — the k=10 preview reproduction attempt was killed mid-run and has not been redone (§7.4, §10).
- Per-topology-size and per-specific-node error analysis (not just per-role) has not been done for either the k=5 or k=10-preview model.
- The multi-task fault-type auxiliary head was tried once (neutral/slightly negative) at an earlier point in the ablation sequence (§4.3, entry 7) but was never re-tried in combination with the later winning changes (SAGE, embedding variance, attention pooling, ensembling) — an open combination not yet tested.

**Documentation:**
- `docs/model/model.md` §8's own internal experiment log is stale (stops at OOD top-1=0.275, calls 0.6 "the target") relative to the real `metrics_history.jsonl` — flagged repeatedly but not yet corrected in that file itself.
- `docs/handoff/project-context-2026-09-27.md` has not been updated with the §7 k=10 preview information in this file.
- Related Work / Literature Review section of the report is blocked on the user supplying the source research papers.
- A proper prose Results section, a dedicated Discussion/Limitations section, and References have not been written yet (§8).
