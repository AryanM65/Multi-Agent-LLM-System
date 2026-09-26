# Infrastructure Setup — Running Dataset Generation on Kaggle & Lightning AI

> How to actually launch `scripts/run_study_vllm.py` (or any GPU-batched generation run) on a free/cheap cloud GPU, using the two platforms this project has used: **Kaggle** (kernels) and **Lightning AI** (Studios + Jobs). Every practice and error below was hit and resolved in real generation runs — treat this as the checklist before launching a new one, not just a read-once doc.

---

## 0. Which platform to use, and why both

- **Kaggle**: free T4 GPU, ~30h/week quota, kernel-based (push code + a dataset, run a script end-to-end, pull logs/output afterward). Proven, used for the original 493-record and expanded ~950-record dataset runs.
- **Lightning AI**: free-tier T4 GPU via "Studios" + "Jobs" (scriptable via the `lightning_sdk` Python package). Used to run a **second, fully parallel** generation job on a disjoint question set, doubling effective throughput without waiting on Kaggle's queue or quota.

Running both at once (on genuinely disjoint data — different questions, so no overlap/duplication risk) is a legitimate way to scale dataset generation faster than either platform's solo quota allows, as long as you're careful about consistency (same model, same code, same fixes on both sides — see §5).

---

## 1. Kaggle — Kernel-Based Generation

### 1.1 Authentication
```bash
pip install kaggle
kaggle auth login --force   # opens a browser OAuth flow
```
**Known gotcha**: if you get `Authentication required to call the Kaggle API` even though `kaggle auth print-access-token` returns a token, the token has gone stale after sitting idle for hours — re-run `auth login --force`.

**PATH gotcha (Windows)**: the `kaggle` CLI script may land in `%APPDATA%\Roaming\Python\Python3xx\Scripts`, which isn't always on PATH by default — add it permanently if `python -m kaggle ...` works but bare `kaggle ...` doesn't.

### 1.2 Code delivery pattern: Dataset + Kernel, not a Notebook
Package your code (`src/`, `scripts/`, `data/`, `dataset/topology_pool.json`) as a private **Kaggle Dataset** (a file-storage primitive, unrelated to ML datasets), then reference it from a separate **Kernel** (script type, GPU + internet enabled). This decouples "iterate on code" (push a new Dataset version) from "manage a compute run" (launch/monitor/kill a Kernel) — much cleaner than embedding code directly in a Notebook.

```bash
cd <staged_package_dir>   # containing src/, scripts/, data/, dataset/
kaggle datasets version -p . --dir-mode zip -m "<message>"
```

**Critical gotcha — verify every push actually applied**: `kaggle datasets version` can print "Upload successful" for every file **without the new content actually going live**. This happened at least once. Always verify before trusting it:
```bash
kaggle datasets files <owner>/<dataset-slug>
# check file sizes/timestamps match what you just pushed, especially for any file whose byte size changed
```
Do this every single time, not just when something seems wrong — the false-positive doesn't announce itself.

### 1.3 Kernel-mount path
A Kaggle Dataset attached to a Kernel mounts at:
```
/kaggle/input/datasets/<owner>/<dataset-slug>
```
**not** the shorter `/kaggle/input/<dataset-slug>` you might expect from Kaggle's own docs examples. Verify with a quick `os.listdir()` debug print at the top of the kernel entry script before assuming paths are wrong elsewhere.

### 1.4 `scripts/` isn't a package on Kaggle
Add `CODE_ROOT/scripts` to `sys.path` and import script modules as top-level (`from run_study2 import X`), not `from scripts.run_study2 import X` — there's no `scripts/__init__.py` convention enforced identically to a local checkout.

### 1.5 Real-time log streaming (critical — do this or a live run looks hung)
```python
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
for line in proc.stdout:
    print(f"[{time.strftime('%H:%M:%S')}] {line.rstrip()}", flush=True)
proc.wait()
```
**Do NOT use** `subprocess.run(capture_output=True)` for the main generation call — it buffers all child output until the process fully exits, making a multi-hour run indistinguishable from a hung one. This mistake cost ~1 hour of confused waiting on an early run before being diagnosed.

To watch a running kernel's logs:
```bash
export PYTHONUTF8=1   # avoids a Windows console-encoding crash on em-dashes/Unicode in log content
timeout 130 kaggle kernels logs -f <owner>/<kernel-slug> > /tmp/log.txt 2>&1
grep -E "=== |records written" /tmp/log.txt
```
`kernels logs -f` **always replays from the start** of the log, so a short `timeout` window may miss recent lines on a long-running kernel — budget ~130s+ to reach recent output, or just re-run it periodically.

### 1.6 Checking kernel status
```bash
kaggle kernels status <owner>/<kernel-slug>
```
No clean "pause"/"stop" command exists — only `kernels delete`, which removes the kernel object entirely (destructive, but re-pushing recreates it). **Always use a fresh, distinctly-named kernel ID/title per launch** rather than reusing an old name — reusing a name risks confusing a stale cached kernel page with a new run.

### 1.7 `--resume` for multi-hour runs
Add topology-level resume granularity to any long generation script:
```python
parser.add_argument("--resume", action="store_true",
    help="Skip topologies already complete in --log-path (topology-level granularity)")
```
A killed/restarted run then redoes at most the one topology that was in progress, not everything. Cheap insurance for anything estimated at several hours.

### 1.8 Real timing data (for ETA estimation)
Per-topology generation time correlates with **how many Reasoner-role nodes run in the same batched round**, not simple node count or chain depth (Reasoner uses a larger token budget and longer chain-of-thought):

| Reasoner nodes in same round | Typical duration |
|---|---|
| 1 | ~7-9 min |
| 2 | ~9-11 min |
| 3 (sequential) | ~12 min |
| 3 (parallel, same round) | ~22 min (worst case) |
| 4 (parallel) | ~21 min |

Model load is a one-time ~8-10 min cost at kernel start, not per-topology.

---

## 2. Lightning AI — Studio + Job-Based Generation

### 2.1 Setup
```bash
pip install lightning-sdk
```
Get an API key from **lightning.ai → Settings → Keys**, and your **username** (not the numeric User ID shown elsewhere on the same page — the SDK needs the username string to resolve your account).

```python
import os
os.environ["LIGHTNING_API_KEY"] = "<key>"
os.environ["LIGHTNING_USERNAME"] = "<username>"   # NOT the UUID-style user ID
```

**Never commit the API key.** Store it in a gitignored JSON file (e.g. `lightning_credentials.json`) and read it in code — same handling as any other credential.

### 2.2 Resolving your teamspace
`User().teamspaces` gives the list, but a personal account's actual GPU-billing teamspace is often **owned by an auto-created organization**, not directly by the user:
```python
from lightning_sdk import User, Organization
u = User()
print(u.teamspaces)          # may show e.g. ['general'] under the user directly
# but Studio(..., teamspace='general', user=...) can 403 / "does not exist"
# — the real teamspace often lives under an org:
for org_name in u.organizations if hasattr(u, "organizations") else []:
    ...
```
In practice: try `Studio(name=..., teamspace=<name>, org=<name>-org, create_ok=True)` — Lightning's own error messages are informative here (e.g. *"Is `general` an organizational Teamspace? You are a member of: ['yourname-org']"*) and will tell you which owner type to use. If a plain `teamspace=` guess 403s or 404s, list `Organization(<org-name>).teamspaces` to find the real name (commonly `default-project`, not the display name shown in the UI).

### 2.3 Creating a Studio and uploading code
```python
from lightning_sdk import Studio
studio = Studio(name="btp-dataset-gen", teamspace="default-project",
                 org="yourname-org", create_ok=True)
```

**Critical Windows bug — do not use `Studio.upload_folder` from a Windows client.** Its internal `Studio.upload_file` calls `os.path.normpath(remote_path)`, which on Windows converts `/` to `\`, so a remote path like `btp_code/data/foo.json` gets silently mangled into a **flat file literally named** `btp_code\data\foo.json` (backslash as a literal character, not a directory separator) on the Linux remote. The Job then fails with `cd: btp_code: No such file or directory` even though `list_artifacts`/`ls` shows files "under" that path.

**Workaround**: bypass the buggy wrapper and call the lower-level API directly with a POSIX path, uploading file-by-file:
```python
import os
for dirpath, dirnames, filenames in os.walk(local_root):
    for fn in filenames:
        local_file = os.path.join(dirpath, fn)
        rel = os.path.relpath(local_file, local_root).replace(os.sep, "/")
        remote_path = "btp_code/" + rel
        studio._studio_api.upload_file(
            studio_id=studio._studio.id, teamspace_id=studio._teamspace.id,
            cloud_account=studio._studio.cluster_id,
            file_path=local_file, remote_path=remote_path, progress_bar=False,
        )
```
Verify afterward with `studio.run("find btp_code -type f | sort")` (requires the Studio to be `Running` — call `studio.start()` first if `Stopped`) — confirm real nested paths, not flat backslash-named files.

**Also avoid** empty (0-byte) files in the uploaded tree (e.g. a bare `src/__init__.py`) — they were observed causing an HTTP 501 mid-upload for `upload_folder`; give every `__init__.py` at least one line (`# package marker`).

### 2.4 Launching a generation Job
```python
from lightning_sdk import Machine
cmd = (
    "cd btp_code && "
    "pip install -q -r requirements.txt && "
    "pip install -q vllm && "
    "pip install -q --upgrade --force-reinstall --no-cache-dir scipy numba mistral-common matplotlib && "
    "mkdir -p logs && "
    "BTP_BACKEND=vllm python -u scripts/run_study_vllm.py "
    "--topology-pool dataset/topology_pool.json "
    "--examples-json data/my_questions.json "
    "--k 5 --batch-size 15 "
    "--examples-per-topology-full-grid 4 --examples-per-topology-partial 7 "
    "--log-path logs/dataset_trials.jsonl --skip-log-path logs/dataset_skipped.jsonl "
    "--resume"
)
job = studio.run_job(name="btp-dataset-gen-run1", machine=Machine.T4, command=cmd)
```

### 2.5 The NumPy 1.x/2.x ABI whack-a-mole (the big one — expect this every time)

**Root cause**: Lightning's base "cloudspace" conda environment ships several packages pre-compiled against **NumPy 1.x** (`scipy`, `matplotlib`, `numba`, and transitively `torchmetrics`/`lightning.pytorch`, which the environment's own `sitecustomize.py` auto-imports on **every single Python invocation** — including your own script). Installing `vllm` silently upgrades NumPy to 2.x as a dependency, without upgrading those other packages — so the very first `import scipy.signal` (or anything importing `torchmetrics`) crashes with:
```
ImportError: A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x as it may crash...
```

**This is not a one-shot fix — each fix uncovers the next broken package** in the same import chain (`torchmetrics → scipy.signal` first, then `→ matplotlib`, then `numba` version-pin conflicts with `vllm`'s own pin). The fix that actually worked, after iterating through the chain:

```bash
pip install -q -r requirements.txt
pip install -q vllm
pip install -q --upgrade --force-reinstall --no-cache-dir scipy numba mistral-common matplotlib
```

Key details that made prior attempts fail silently:
- `--force-reinstall` **alone, without `--upgrade`**, just reinstalls whatever version pip already resolved — it does **not** guarantee a NumPy-2-compatible version. You need both flags together.
- `pip check` after each install step surfaces exactly which packages still declare a `numpy<2` requirement — use it to find the full list instead of discovering them one crash at a time:
  ```bash
  pip check 2>&1 | grep numpy
  ```
- A `vllm==X` version may pin `numba==<exact-version>` too; bumping numba for NumPy-2 compatibility can conflict with that pin. In practice this conflict was **cosmetic** (a pip warning, not a runtime failure) — verify by directly testing the actual import chain end-to-end (below) rather than trusting `pip check`'s conflict warnings alone.

**Debug workflow that actually works** (much faster than relaunching a full Job each time, which costs ~5-10 min of machine-setup overhead per attempt): install and test directly on the **Studio's own persistent shell**, which is separate from any Job's isolated snapshot:
```python
studio.start()   # if Stopped
out = studio.run("pip install ...")          # returns combined stdout+stderr
out = studio.run("pip check 2>&1")
```
Then verify the actual import chain with an uploaded test script (avoid inline `python -c "..."` through `studio.run()` — multi-layer shell quoting through Python→shell→remote-python reliably mangles nested quotes/escapes):
```python
# write test_imports.py locally: import scipy.signal; import matplotlib; import numba; from vllm import SamplingParams; import src.faults
studio.upload_file(local_path, remote_path="btp_code/test_imports.py")
studio.run("cd btp_code && python3 test_imports.py")
```
Only relaunch the full Job once this passes cleanly.

### 2.6 Concurrent GPU capacity limit
A Job whose internal script crashes does **not** automatically stop the underlying machine — `job.status` can stay `Running` indefinitely even after the process inside has exited with an error. If you relaunch without stopping the old one, the new Job can get stuck in `Pending` forever, silently blocked by the account's concurrent-GPU-machine limit (observed: 2 concurrent T4 machines on a free-tier account). Symptom: `job.started_at is None` after many minutes.

**Always explicitly stop dead/crashed jobs before relaunching**:
```python
from lightning_sdk import Job
Job(name="old-run-name", teamspace=..., org=...).stop()
```
Check `job.status` across *all* recent launch attempts, not just the latest one, when a new launch seems stuck.

### 2.7 Retrieving generated output mid-run or after completion
A Job's working directory is an isolated snapshot taken at launch — **not** live-shared with the Studio's persistent storage (files a Job creates while running, e.g. `logs/dataset_trials.jsonl`, are invisible via `studio.run("ls ...")`). Use the Job's own artifact API instead:
```python
job.list_artifacts(path="btp_code/logs")        # confirms the file exists + size, before download
job.download_artifacts(target_dir="local_dir", path="btp_code/logs/dataset_trials.jsonl")
```
**Known quirk**: `list_artifacts` can show a file (with a real, growing size) while `download_artifacts` reports `"No files found"` for the same path — an eventual-consistency lag in the artifact indexer for files a *still-running* Job is actively writing to. If this happens, either wait for the Job to progress further (next topology boundary) or for it to fully complete, then retry the download — don't assume the file is actually missing.

### 2.8 Monitoring a running Job
```python
job.status                         # 'Pending' -> 'Running' -> 'Completed'/'Failed'/'Stopped'
job.logs()                         # raises RuntimeError while status == 'Pending' -- wrap in try/except
job.machine                        # confirms which GPU actually got allocated (e.g. 'T4')
```
Poll in a loop with a fixed sleep interval; don't assume `Pending` means stuck within the first few minutes — machine provisioning + Studio snapshotting routinely takes 1-3 minutes before a Job transitions to `Running`.

---

## 3. Shared practices (apply to both platforms)

### 3.1 Verify a data-quality claim by inspecting actual content, not just "did it run"
The single most valuable practice from this project's history: a full generation run can complete with **zero crashes and correct-looking record counts** while still containing real correctness bugs (e.g. a fault type silently not being applied to the prompt at all). These only surfaced when someone actually opened generated records and checked whether the *labeled* fault visibly changed the output. Always do a manual content spot-check — pick a few records per fault type, read the actual `samples` field — before trusting a generation run's output, on either platform.

### 3.2 Back up before regenerating
Before replacing a dataset file with a new run's output, copy the old one aside (e.g. `trials_v2_493.jsonl`) rather than overwriting in place — regeneration runs are expensive (hours of GPU time), and a bug discovered in the new run shouldn't cost you the old, working dataset too.

### 3.3 Keep runs on genuinely disjoint data when parallelizing across platforms
If running Kaggle and Lightning AI (or any two compute sources) in parallel to increase total dataset volume, make sure each draws from a **disjoint** question set — verify this explicitly in code (`assert not (set_a & set_b)`) before launching, not after. Otherwise you end up with duplicated (question, topology, fault) recipes across the two outputs, which complicates train/val splitting and wastes compute re-deriving information you already have.

### 3.4 Same code, same model, on every parallel run
When splitting a generation run across platforms, push the exact same fixed code (and note the exact commit/version) to both — any bug fix or config change made on one side must be mirrored on the other before both are considered part of the same "dataset version," or the combined output will be internally inconsistent (part of it carrying a bug the other part doesn't).

---

## 4. Quick-reference command cheatsheet

```bash
# --- Kaggle ---
kaggle auth login --force
kaggle datasets version -p <staged_dir> --dir-mode zip -m "message"
kaggle datasets files <owner>/<slug>                      # verify push applied
kaggle kernels push -p <kernel_dir>                       # launch/relaunch
kaggle kernels status <owner>/<kernel-slug>
PYTHONUTF8=1 timeout 130 kaggle kernels logs -f <owner>/<kernel-slug>
kaggle kernels output <owner>/<kernel-slug> -p <local_dir>  # pull final output
kaggle kernels delete <owner>/<kernel-slug>               # only way to "stop"

# --- Lightning AI (Python, not CLI) ---
# see full snippets in Section 2 above
```

## 5. Open items / things to double check on the next run

- Whether the `numba` version bump (past `vllm`'s pinned exact version) ever causes a *runtime* (not just import-time) behavior difference in vLLM's quantization kernels — not fully ruled out, only import-level testing was done.
- Lightning AI's free-tier GPU-hour budget isn't as clearly surfaced as Kaggle's 30h/week — check the account's current balance before relying on it for a many-hour run.
- Consider scripting the "stop all Running jobs older than N minutes with no `started_at`" check as a guard, since the concurrent-capacity deadlock (§2.6) is easy to hit again after several relaunch iterations.
