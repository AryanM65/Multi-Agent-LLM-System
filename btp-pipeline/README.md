# btp-pipeline

Core library and scripts for the multi-agent LLM uncertainty-propagation pipeline.

- **Project overview, research motivation, architecture**: see the [root README](../README.md).
- **All other documentation, organized by topic**: see [`../docs/`](../docs/README.md) — dataset schema, GNN model plan, Kaggle/Lightning AI infra setup, pipeline internals reference, local dev guide, and standalone-agent usage.

Quick local setup:
```bash
python -m venv btp-env
source btp-env/bin/activate   # Windows: btp-env\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py   # one-time, ~50MB HotpotQA cache
```
For the full local-development workflow (sanity checks, scored runs, mock mode), see [`docs/pipeline/local-dev-guide.md`](../docs/pipeline/local-dev-guide.md).
