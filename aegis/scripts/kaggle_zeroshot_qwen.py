"""Phase 3a: zero-shot Qwen2.5-7B-Instruct (AWQ) on Aegis-Bench test + Who&When, via vLLM on Kaggle.

Reproduces the paper's Table 1 row "Qwen2.5-7B-Instruct (zero-shot)" using the authors' own
prompt (evaluation/prompt.txt), system message and conversation formatting (local_model.py).
Deviations from the paper (document in docs/reproduction_log.md): AWQ 4-bit weights instead of
full precision, greedy decoding (temperature 0), and head+tail truncation when a prompt exceeds
the context budget (the authors do not truncate).

Inputs (found by file name anywhere under --input-dir, default /kaggle/input):
  test.jsonl, whowhen.jsonl, prompt.txt
Outputs (in --output-dir, default /kaggle/working), resumable and appended per batch:
  zeroshot_test.jsonl, zeroshot_who_when.jsonl, run_info.json
Each output line: id, raw_output, prompt_tokens, truncated, finish_reason.
Parsing and scoring happen locally (aegis/src/metrics.py), so raw outputs are kept untouched.

Never reads ground truth: only metadata-free fields input.query and input.conversation_history
(step, agent_name, content) are used to build the prompt.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

SYSTEM_MSG = (
    "You are a precise analyst. Please follow the instructions carefully and provide detailed "
    "reasoning before giving your final JSON answer."
)


def find_file(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def build_conversation_text(rec: dict) -> str:
    """Identical to the authors' extract_conversation_text (local_model.py)."""
    inp = rec["input"]
    text = f"QUERY:\n{inp.get('query', '')}\n\nCONVERSATION HISTORY:\n"
    for e in inp.get("conversation_history", []):
        text += f"Step {e.get('step', '')} - {e.get('agent_name', '')}:\n{e.get('content', '')}\n\n"
    return text.strip()[:100000]


def truncate_middle(tok, text: str, max_tokens: int, head_frac: float = 0.3) -> str:
    """Keep the first head_frac and last (1-head_frac) of the tokens, joined by a marker."""
    ids = tok(text, add_special_tokens=False).input_ids
    if len(ids) <= max_tokens:
        return text
    h = int(max_tokens * head_frac)
    t = max_tokens - h
    return tok.decode(ids[:h]) + "\n\n[... middle of conversation omitted for length ...]\n\n" + tok.decode(ids[-t:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="/kaggle/input")
    ap.add_argument("--output-dir", default="/kaggle/working")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32, help="records per resumable chunk")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N records per split")
    ap.add_argument("--splits", default="test,who_when")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size (2 for 2xT4)")
    # Authors'-setup options (defaults keep the original Phase 3a behaviour: AWQ, greedy, HF test file).
    ap.add_argument("--quant", default="awq", choices=["awq", "none"], help="none = full-precision fp16 weights")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--rep-penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-name", default="test.jsonl", help="file name of the test split (aegis.jsonl = the repo's other 600)")
    ap.add_argument("--subset", type=int, default=0, help="random subset of N records per split (seeded by --subset-seed)")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    inp, out = Path(args.input_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prompt_tpl = find_file(inp, "prompt.txt").read_text(encoding="utf-8")
    files = {"test": find_file(inp, args.test_name), "who_when": find_file(inp, "whowhen.jsonl")}

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model, quantization=("awq" if args.quant == "awq" else None), dtype="float16",
        max_model_len=args.max_model_len, gpu_memory_utilization=0.90, tensor_parallel_size=args.tp, enforce_eager=True,
    )
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                        repetition_penalty=args.rep_penalty, seed=(args.seed if args.temperature > 0 else None),
                        max_tokens=args.max_new_tokens)

    # Budget for the conversation text: context minus generation, minus the fixed prompt/system overhead.
    overhead = len(tok(prompt_tpl.format(conversation_text="") + SYSTEM_MSG).input_ids) + 64
    conv_budget = args.max_model_len - args.max_new_tokens - overhead
    info = {"model": args.model, "max_model_len": args.max_model_len, "conv_token_budget": conv_budget,
            "max_new_tokens": args.max_new_tokens, "quant": args.quant, "temperature": args.temperature,
            "top_p": args.top_p, "top_k": args.top_k, "rep_penalty": args.rep_penalty, "seed": args.seed,
            "test_name": args.test_name, "subset": args.subset, "subset_seed": args.subset_seed, "splits": {}}

    for split in args.splits.split(","):
        recs = [json.loads(l) for l in files[split].read_text(encoding="utf-8").splitlines() if l.strip()]
        if args.subset:
            import random
            recs = random.Random(args.subset_seed).sample(sorted(recs, key=lambda r: r["id"]), min(args.subset, len(recs)))
        if args.limit:
            recs = recs[: args.limit]
        out_path = out / f"zeroshot_{split}{args.tag}.jsonl"
        done = set()
        if out_path.exists():
            done = {json.loads(l)["id"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        todo = [r for r in recs if r["id"] not in done]
        print(f"[{split}] {len(recs)} records, {len(done)} done, {len(todo)} to run", flush=True)
        n_trunc = 0
        t0 = time.time()
        for i in range(0, len(todo), args.batch):
            chunk = todo[i: i + args.batch]
            prompts, meta = [], []
            for r in chunk:
                conv = build_conversation_text(r)
                cut = truncate_middle(tok, conv, conv_budget)
                truncated = cut is not conv and cut != conv
                n_trunc += truncated
                user = prompt_tpl.format(conversation_text=cut)
                text = tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": user}],
                    tokenize=False, add_generation_prompt=True,
                )
                prompts.append(text)
                meta.append((r["id"], truncated, len(tok(text).input_ids)))
            outs = llm.generate(prompts, sp)
            with out_path.open("a", encoding="utf-8") as f:
                for (rid, trunc, ntok), o in zip(meta, outs):
                    f.write(json.dumps({
                        "id": rid, "raw_output": o.outputs[0].text, "prompt_tokens": ntok,
                        "truncated": trunc, "finish_reason": o.outputs[0].finish_reason,
                    }, ensure_ascii=False) + "\n")
            print(f"[{split}] {min(i + args.batch, len(todo))}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
        info["splits"][split] = {"records": len(recs), "newly_run": len(todo), "truncated_this_run": n_trunc,
                                 "seconds": round(time.time() - t0)}
    (out / f"run_info{args.tag}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
