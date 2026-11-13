"""LoRA small-LM reader for failure attribution (Kaggle T4). Standalone.

A 1.5B-class instruction model reads the query and the whole conversation and answers with the faulty agents and their error modes,
one line per agent: "<agent name>: FM-x.y, FM-z.w". Trained on AEGIS GROUPED train only (LoRA, causal-LM loss on the answer tokens);
AEGIS test and Who&When are generated once at the end (zero-shot for Who&When, like the paper's Aegis-SFT). No Who&When label is used.

Inputs under --input-dir: train.jsonl, val.jsonl, test.jsonl, whowhen.jsonl, grouped_ids.json (same datasets as the encoder kernels).
Outputs (/kaggle/working): preds_<split>.json  {record id: [[agent, mode], ...]},  lora_log.json,  adapter/ (LoRA weights).
Score on CPU with aegis/src/eval/score_lora.py.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)   # Kaggle image ships a torchao that peft rejects

import numpy as np
import torch

MODES = {
    "FM-1.1": "Disobey task specification", "FM-1.2": "Disobey role specification", "FM-1.3": "Step repetition",
    "FM-1.4": "Loss of conversation history", "FM-1.5": "Unaware of termination conditions",
    "FM-2.1": "Conversation reset", "FM-2.2": "Fail to ask for clarification", "FM-2.3": "Task derailment",
    "FM-2.4": "Information withholding", "FM-2.5": "Ignored other agent's input", "FM-2.6": "Reasoning-action mismatch",
    "FM-3.1": "Premature termination", "FM-3.2": "No or incorrect verification", "FM-3.3": "Weak verification",
}
SYSTEM = ("You analyse failed multi-agent LLM system runs. Some agents behaved faultily. Error modes: "
          + "; ".join(f"{k} {v}" for k, v in MODES.items())
          + ". Answer with one line per faulty agent, exactly '<agent name>: <mode>, <mode>' using only these codes.")


def norm_mode(c: str) -> str:
    return str(c).strip().upper().replace("EM-", "FM-").replace("FM_", "FM-")


def find(root: Path, name: str) -> Path:
    return sorted(root.rglob(name))[0]


def read(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def prompt_text(rec: dict, char_budget: int = 7500) -> str:
    turns = rec["input"]["conversation_history"]
    per = max(150, min(700, char_budget // max(len(turns), 1)))
    lines = [f"Task: {(rec['input']['query'] or '')[:600]}", f"Framework: {rec['metadata']['framework']}", "Conversation:"]
    for t in turns:
        lines.append(f"[{t.get('step', '')}] {str(t['agent_name']).strip()}: {(t.get('content') or '')[:per]}")
    return "\n".join(lines)


def answer_text(rec: dict) -> str:
    by: dict = {}
    for a in rec.get("output", {}).get("faulty_agents", []):
        by.setdefault(str(a["agent_name"]).strip(), set()).add(norm_mode(a["error_type"]))
    return "\n".join(f"{ag}: {', '.join(sorted(ms))}" for ag, ms in sorted(by.items()))


def parse_answer(text: str, agents: set) -> list:
    out = []
    for line in text.strip().splitlines():
        if ":" not in line:
            continue
        ag, rest = line.rsplit(":", 1)
        ag = ag.strip()
        modes = re.findall(r"FM-\d\.\d", rest)
        if ag in agents:
            out += [[ag, m] for m in dict.fromkeys(modes) if m in MODES]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="/kaggle/input")
    ap.add_argument("--output-dir", default="/kaggle/working")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train-n", type=int, default=1600, help="training records (random subset of grouped train)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-prompt-tokens", type=int, default=2300)
    ap.add_argument("--train-minutes", type=float, default=170.0, help="stop training after this many minutes and go on to generation")
    ap.add_argument("--val-n", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: tiny run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    root, out = Path(args.input_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(dev)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    ids = json.loads(find(root, "grouped_ids.json").read_text())
    recs = {}
    for name in ("train.jsonl", "val.jsonl"):
        for r in read(find(root, name)):
            recs[r["id"]] = r
    train = [recs[i] for i in ids["train"] if i in recs]
    val = [recs[i] for i in ids["val"] if i in recs][: args.val_n]
    random.Random(args.seed).shuffle(train)
    train = train[: (8 if args.limit else args.train_n)]

    def encode(rec, with_answer=True):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt_text(rec)}]
        p_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        if hasattr(p_ids, "input_ids"):
            p_ids = p_ids["input_ids"]
        p_ids = list(p_ids)
        if len(p_ids) > args.max_prompt_tokens:   # keep the head (task + role names) and the tail (end of conversation)
            keep = args.max_prompt_tokens
            p_ids = p_ids[: keep // 3] + p_ids[-(keep - keep // 3):]
        if not with_answer:
            return p_ids
        a_ids = tok(answer_text(rec) + tok.eos_token, add_special_tokens=False)["input_ids"]
        return p_ids, a_ids

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    log, t0, step, tot, nb = [], time.time(), 0, 0.0, 0
    model.train()
    stop = False
    for ep in range(args.epochs):
        random.Random(args.seed + ep).shuffle(train)
        for rec in train:
            p_ids, a_ids = encode(rec)
            ids_t = torch.tensor([p_ids + a_ids], device=dev)
            labels = torch.tensor([[-100] * len(p_ids) + a_ids], device=dev)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids_t, labels=labels).loss / args.accum
            scaler.scale(loss).backward()
            tot += loss.item() * args.accum; nb += 1
            if nb % args.accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(); step += 1
                if step % 10 == 0:
                    print(f"step {step} loss {tot / nb:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
                    log.append({"step": step, "loss": tot / nb, "minutes": (time.time() - t0) / 60})
                    (out / "lora_log.json").write_text(json.dumps(log, indent=1))
            if (time.time() - t0) / 60 > args.train_minutes:
                print("training time budget reached", flush=True)
                stop = True
                break
        if stop:
            break
    model.save_pretrained(out / "adapter")
    model.eval()

    @torch.no_grad()
    def generate(rows, name, bs=4):
        res, order = {}, sorted(range(len(rows)), key=lambda i: len(rows[i]["input"]["conversation_history"]))
        for b in range(0, len(order), bs):
            batch = [rows[i] for i in order[b: b + bs]]
            enc = [encode(r, with_answer=False) for r in batch]
            L = max(len(e) for e in enc)
            inp = torch.tensor([[tok.pad_token_id] * (L - len(e)) + e for e in enc], device=dev)
            att = torch.tensor([[0] * (L - len(e)) + [1] * len(e) for e in enc], device=dev)
            with torch.autocast("cuda", dtype=torch.float16):
                gen = model.generate(input_ids=inp, attention_mask=att, max_new_tokens=70, do_sample=False, pad_token_id=tok.pad_token_id)
            for r, g in zip(batch, gen):
                txt = tok.decode(g[L:], skip_special_tokens=True)
                agents = {str(t["agent_name"]).strip() for t in r["input"]["conversation_history"]}
                res[r["id"]] = {"pairs": parse_answer(txt, agents), "raw": txt}
            if (b // bs) % 25 == 0:
                print(f"  {name}: {b + len(batch)}/{len(rows)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        (out / f"preds_{name}.json").write_text(json.dumps(res))
        return res

    test = list(read(find(root, "test.jsonl")))
    ww = list(read(find(root, "whowhen.jsonl")))
    if args.limit:
        test, ww, val = test[:8], ww[:8], val[:8]
    generate(ww, "who_when")      # most important first, in case time runs out
    generate(test, "test")
    generate(val, "val")
    (out / "run_info.json").write_text(json.dumps({"model": args.model, "train_n": len(train), "steps": step, "minutes": (time.time() - t0) / 60}))
    print("done", flush=True)


if __name__ == "__main__":
    main()
