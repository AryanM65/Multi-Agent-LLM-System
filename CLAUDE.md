# Project instructions

- Permissions: maximal autonomy for commands (commits, pushes, installs run without asking) AND for file writes/edits (Edit/Write tool run without asking, per explicit user instruction 2026-09-28). Still always ask first for: force push, reset --hard, branch delete, rm -rf, anything touching credentials/secrets files. See .claude/settings.json for exact allow/ask lists.

- **Always keep logs — never let them be ephemeral.** Kaggle kernel logs are NOT readable while a kernel is RUNNING (confirmed repeatedly: `kaggle kernels logs`/`kernels output` both return empty until COMPLETE) and are lost forever if the kernel is deleted before COMPLETE (this cost a full dataset-gen run's worth of data once already). Rules:
  - The moment ANY kernel (or any long-running job) reaches COMPLETE, pull its full log/output and save it into the repo (or another durable location) before doing anything else with it — don't just read it and move on.
  - Never delete/stop a RUNNING kernel unless its output has already been pulled and saved, or the loss is explicitly acceptable and confirmed. If a kernel must be interrupted, prefer batching it into smaller resumable chunks (via `--topology-limit`-style flags or equivalent) specifically so there's always a recent completed checkpoint to fall back on.
  - Per-epoch / per-run training logs are real research artifacts, not scratch files — do not delete them during "cleanup" passes. If unsure whether a log is disposable, keep it or ask first.
  - When generating data or training across multiple parallel jobs, back up each one's output to the repo (e.g. `dataset/k10_partial/`) as soon as it lands, rather than waiting to collect everything at the end.
