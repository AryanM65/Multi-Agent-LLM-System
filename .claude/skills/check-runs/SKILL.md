---
name: check-runs
description: Check status of both dataset-generation runs (Kaggle + Lightning AI) and report progress
---

Check status of both dataset-generation runs:

1. **Kaggle** kernel `aryanmahajan7/btp-dataset-gen-v3-expanded` — via `python -m kaggle kernels status aryanmahajan7/btp-dataset-gen-v3-expanded`, then tail the kernel log for topology/record progress (grep for topology names / record counts in recent log lines).

2. **Lightning AI** job `btp-dataset-gen-run6` — via `lightning_sdk`, using credentials in `lightning_credentials.json` (root of repo: `LIGHTNING_API_KEY`, `LIGHTNING_USERNAME`, `teamspace: default-project`, `org: aryanmahajan600-org`). Wrap `job.logs()` in try/except (raises `RuntimeError` while status is `Pending`). Tail logs for topology/record progress.

Report:
- Current status (Running / Complete / Failed / Stopped) for each.
- Topology progress (e.g. "17/20 topologies, ~700 records") if determinable from logs.
- Flag clearly if either reached COMPLETE/Failed/Stopped.

Keep the report short — status + progress numbers, not full log dumps.
