"""Frozen-LLM judge for failure attribution candidates (Kaggle T4, or CPU for smoke tests). Standalone, nothing is trained.

For every trajectory it reads the shortlisted (agent, mode) candidates written by aegis/src/eval/rr_candidates.py and asks a frozen instruction model one
yes/no question per candidate: "did agent A commit error FM-x.y somewhere in this conversation?". The score is a log-odds
    J = log P(Yes) - log P(No)    (read from ONE next-token distribution, no free generation)
Order-bias control (--format ab, default): the answer options are letters; the question is asked twice with the option order swapped
(A=Yes,B=No and A=No,B=Yes) and J is the average of the two log-odds (the two raw values are kept in the output).
Efficiency: the conversation is the shared prefix of all k candidates of a record, so it is run once and its key/value cache is reused for every
candidate suffix (cache.crop back to the prefix length after each candidate).

Variants (--variant):
  plain : instruction + the paper's 14 mode definitions (aegis/third_party/AEGIS/evaluation/prompt.txt wording) + conversation + the candidate question
  rag   : same, plus the candidate's mode-first retrieved AEGIS-train examples WITH their own ground-truth labels (field examples_rag)
  rand  : same format as rag but random AEGIS-train examples of the same kind (field examples_rand); the control for the retrieval claim
Candidates without examples (weak-match fallback) are asked exactly like plain.

Inputs (found anywhere under --input-dir, several allowed): cand_<tag>_<split>.jsonl (from rr_candidates.py) and the raw AEGIS / Who&When jsonl files
(train.jsonl, val.jsonl, test.jsonl, whowhen.jsonl, the same files the encoder / LoRA kernels use) to get the conversations by id.
Output (--output-dir): judge_<variant>_<split>.jsonl, one line per trajectory, written incrementally (a kernel timeout loses at most one record);
run_info.json. Score on CPU with aegis/src/eval/rr_fuse.py. Who&When is only judged here, never used to tune anything.

Kaggle example: python kaggle_judge.py --input-dir /kaggle/input --cands cand_rr_base_val.jsonl --variant rag --model Qwen/Qwen2.5-7B-Instruct --four-bit
CPU smoke test: python aegis/scripts/kaggle_judge.py --input-dir aegis/data/raw aegis/third_party/AEGIS/evaluation/data aegis/results/rr \
                --cands cand_rr_smoke_who_when.jsonl --variant rag --model Qwen/Qwen2.5-0.5B-Instruct --device cpu --limit 2 --output-dir <tmp>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 14 mode definitions, verbatim wording of aegis/third_party/AEGIS/evaluation/prompt.txt
DEFS = {
    "FM-1.1": ("Task specification deviation", "Agent deviates from specified task requirements (e.g., was asked to write code in Python, but used JavaScript)."),
    "FM-1.2": ("Role specification deviation", "Agent acts outside its designated role (e.g., a 'CodeWriter' agent starts criticizing other agents' work, which is the 'Critic's' role)."),
    "FM-1.3": ("Add redundant steps", "Agent adds unnecessary or duplicate steps (e.g., imports a library that was already imported in a previous step)."),
    "FM-1.4": ("Remove conversation history", "Agent ignores or removes important context from previous turns (e.g., ignores a user's correction from the previous message)."),
    "FM-1.5": ("Remove termination conditions", "Agent fails to define proper stopping criteria, leading to loops or unfinished tasks (e.g., writes a recursive function with no base case)."),
    "FM-2.1": ("Repeat handled tasks", "Agent redundantly handles already completed tasks (e.g., re-writes a piece of code that was already finalized and approved)."),
    "FM-2.2": ("Make request ambiguous", "Agent provides unclear or confusing instructions to other agents (e.g., asks another agent to \"handle the data\" without specifying how)."),
    "FM-2.3": ("Deviate from main goal", "Agent pursues objectives unrelated to the main task (e.g., starts discussing the history of programming languages in the middle of a coding task)."),
    "FM-2.4": ("Hide important information", "Agent withholds crucial information needed by other agents (e.g., knows a library has a bug but doesn't mention it)."),
    "FM-2.5": ("Ignore other agents", "Agent fails to consider input, corrections, or questions from other agents."),
    "FM-2.6": ("Inconsistent reasoning", "Agent's logic contradicts its own previous statements (e.g., in step 2 agent says 'option A is best', but in step 4 says 'option A is a bad choice' without new information)."),
    "FM-3.1": ("Premature termination", "Agent stops or declares the task complete before all requirements are met."),
    "FM-3.2": ("Remove verification steps", "Agent skips necessary validation or testing steps (e.g., writes code but doesn't write any unit tests for it)."),
    "FM-3.3": ("Incorrect verification", "Agent performs flawed or wrong verification (e.g., writes a test that doesn't actually check for the correct condition)."),
}
SYSTEM = ("You are a meticulous Multi-Agent System (MAS) Quality Assurance analyst. You analyse conversation logs of multi-agent LLM systems and judge, "
          "using strict definitions, whether one specific agent committed one specific error. Answer exactly in the requested format.")
MARK = "<<<CANDIDATE_SPLIT>>>"


def find_all(roots: list, name: str) -> list:
    out = []
    for r in roots:
        out += sorted(Path(r).rglob(name))
    return out


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def conversation_text(rec: dict, char_budget: int) -> str:
    """Same layout as the paper's evaluation prompt ('Step {step} - {agent}:'); long turns are cut so the whole text stays near char_budget."""
    turns = rec["input"]["conversation_history"]
    per = max(300, min(3000, char_budget // max(len(turns), 1)))
    s = f"QUERY:\n{(rec['input']['query'] or '')[:1500]}\n\nCONVERSATION HISTORY:\n"
    for t in turns:
        s += f"Step {t.get('step', '')} - {str(t['agent_name']).strip()}:\n{(t.get('content') or '')[:per]}\n\n"
    return s.strip()


def prefix_user_text(rec: dict, char_budget: int) -> str:
    defs = "\n".join(f"- {c}: **{t}** - {d}" for c, (t, d) in DEFS.items())
    return (f"## ERROR DEFINITIONS WITH EXAMPLES\n{defs}\n\n## CONVERSATION TO ANALYZE:\n\"\"\"\n{conversation_text(rec, char_budget)}\n\"\"\"\n")


def examples_block(exs: list, code: str) -> str:
    if not exs:
        return ""
    s = ("\nBelow are labelled examples of agents from OTHER multi-agent runs (different tasks, possibly a different framework). They only show what this error "
         "looks like or does not look like. They say nothing about the agent in question.\n")
    for i, e in enumerate(exs, 1):
        verdict = f"COMMITTED {code}" if e["label"] else f"did NOT commit {code}"
        s += (f"Example {i} (framework {e['framework']}): agent '{e['agent']}' wrote: {e['snippet']}\n"
              f"  Ground truth: this agent {verdict} (all labelled errors of this agent: {', '.join(e['modes']) or 'none'}).\n")
    return s


def question_text(agent: str, code: str, exs: list, option_yes_first: bool, fmt: str) -> str:
    title, desc = DEFS[code]
    q = (f"\n## QUESTION\nAgent in question: '{agent}'\nError code to check: {code}: **{title}** - {desc}\n{examples_block(exs, code)}\n"
         f"Did the agent '{agent}' commit the error {code} at any point in the conversation above? Judge only this agent and only this error code.\n")
    if fmt == "yn":
        return q + "Answer with a single word: Yes or No."
    yes, no = "Yes, the agent committed this error.", "No, the agent did not commit this error."
    a, b = (yes, no) if option_yes_first else (no, yes)
    return q + f"A) {a}\nB) {b}\nAnswer with a single letter: A or B."


def strip_flag(argv: list, flag: str) -> list:
    """Remove --flag value pairs (and --flag=value) from an argv list, plus the flags the parent overrides for each child."""
    drop = {flag, "--gpu-index", "--shard-base"}
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a.split("=")[0] in drop:
            i += 1 if "=" in a else 2
            continue
        out.append(a)
        i += 1
    return out


def crop_to(cache, L: int) -> None:
    """Cut the key/value cache back to its first L tokens. transformers versions disagree on crop() (newer ones reject a positive length and want a
    negative number = tokens to remove); try both and verify the length, never trust silently."""
    cur = cache.get_seq_length()
    if cur == L:
        return
    try:
        cache.crop(L)
    except ValueError:
        pass
    if cache.get_seq_length() != L:
        cache.crop(-(cur - L))
    if cache.get_seq_length() != L:
        raise RuntimeError(f"could not crop cache to {L} (now {cache.get_seq_length()})")



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", nargs="+", default=["/kaggle/input"])
    ap.add_argument("--output-dir", default="/kaggle/working")
    ap.add_argument("--cands", required=True, help="candidate file name(s), comma separated, e.g. cand_rrc_val.jsonl,cand_rrc_who_when.jsonl (one model load for all)")
    ap.add_argument("--variant", default="plain", help="plain, rag, rand or a comma list (all variants of a record share one conversation prefix, so a list is cheaper than separate runs)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--four-bit", action="store_true", help="bitsandbytes NF4 (needed for 7B-8B on a 16 GB T4)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--format", choices=["ab", "yn"], default="ab")
    ap.add_argument("--char-budget", type=int, default=20000, help="characters of conversation text (about 5-6k tokens)")
    ap.add_argument("--max-prefix-tokens", type=int, default=7000)
    ap.add_argument("--minutes", type=float, default=0.0, help="stop cleanly after this many minutes (0 = no limit)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0, help="skip the first N records (to split one candidate file over several kernels)")
    ap.add_argument("--prefill-chunk", type=int, default=1024, help="tokens per forward pass when reading the shared prefix (memory, not content)")
    ap.add_argument("--gpu-index", type=int, default=0, help="GPU this process uses (4-bit loading pins the model there)")
    ap.add_argument("--shard-base", type=int, default=0, help="first shard id of this process (strided split: record j goes to shard j mod --shard-total)")
    ap.add_argument("--shard-total", type=int, default=0, help="number of shards over ALL processes of ALL kernels (0 = no sharding)")
    ap.add_argument("--spawn", type=int, default=0, help="launch this many child processes of this same script, child k on GPU k, shard --shard-base+k; the parent only waits")
    ap.add_argument("--spawn-delay", type=float, default=150.0, help="seconds between child starts (avoids two simultaneous CPU-RAM peaks while loading a 7B model)")
    args = ap.parse_args()
    if args.spawn:      # parent: start one child per GPU, then wait. The children are this same script, so the parent loads nothing.
        import subprocess
        base = [a for a in sys.argv[1:]]
        kids = []
        for k in range(args.spawn):
            cmd = [sys.executable, os.path.abspath(sys.argv[0])] + strip_flag(base, "--spawn") + ["--gpu-index", str(k), "--shard-base", str(args.shard_base + k)]
            print("spawn", cmd[2:], flush=True)
            kids.append(subprocess.Popen(cmd))
            if k + 1 < args.spawn:
                time.sleep(args.spawn_delay)
        codes = [kd.wait() for kd in kids]
        print("children exit codes", codes, flush=True)
        sys.exit(max(codes))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = [v for v in args.variant.split(",") if v]
    assert all(v in ("plain", "rag", "rand") for v in variants), variants

    tok = AutoTokenizer.from_pretrained(args.model)
    kw: dict = {}
    if args.four_bit:
        from transformers import BitsAndBytesConfig

        kw = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16),
              "device_map": {"": args.gpu_index}}
    else:
        kw = {"torch_dtype": torch.float16 if args.device == "cuda" else torch.float32}
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    if not args.four_bit:
        model.to(args.device)
    model.eval()
    dev = next(model.parameters()).device

    def tid(s: str) -> int:
        return tok.encode(s, add_special_tokens=False)[0]
    if args.format == "ab":
        id_first, id_second = tid("A"), tid("B")
    else:
        id_first, id_second = tid("Yes"), tid("No")

    def render(prefix_user: str, suffix: str) -> tuple:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prefix_user + MARK + suffix}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        pre, post = text.split(MARK)
        return pre, post

    @torch.no_grad()
    def prefill(ids: torch.Tensor):
        """Run the shared conversation prefix through the model in chunks (default 1024 tokens) so the attention buffers stay small: a 7k-token one-shot
        prefill ran out of memory on a 15 GB T4 next to the 4-bit 7B model. Same result as one pass up to float noise."""
        cache, step = None, max(args.prefill_chunk, 1)
        for i in range(0, ids.shape[1], step):
            x = ids[:, i:i + step].to(dev)
            try:
                o = model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1)
            except TypeError:
                o = model(input_ids=x, past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            del o
        return cache

    @torch.no_grad()
    def next_token_logodds(ids: torch.Tensor, cache) -> float:
        try:
            out = model(input_ids=ids.to(dev), past_key_values=cache, use_cache=True, logits_to_keep=1)
        except TypeError:
            out = model(input_ids=ids.to(dev), past_key_values=cache, use_cache=True)
        lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        return float(lp[id_first] - lp[id_second])

    t0 = time.time()
    stop = False
    sfx = f"_s{args.skip}" if args.skip else ""
    if args.shard_total:
        sfx += f"_sh{args.shard_base}of{args.shard_total}"
    for cname in [c for c in args.cands.split(",") if c]:      # several candidate files (e.g. val and who_when) share ONE model load; --minutes is a global budget
        if stop:
            break
        cands = list(read_jsonl(find_all(args.input_dir, cname)[0]))[args.skip:]
        if args.limit:
            cands = cands[: args.limit]
        if args.shard_total:
            cands = cands[args.shard_base:: args.shard_total]
        need = {c["id"] for c in cands}
        recs = {}
        for name in ("train.jsonl", "val.jsonl", "test.jsonl", "whowhen.jsonl"):
            for p in find_all(args.input_dir, name):
                for r in read_jsonl(p):
                    if r["id"] in need:
                        recs[r["id"]] = r
        missing = need - set(recs)
        assert not missing, f"{len(missing)} candidate ids have no raw record (first: {sorted(missing)[:3]})"
        split = cands[0]["split"] if cands else "none"
        fouts = {v: (out_dir / f"judge_{v}_{split}{sfx}.jsonl").open("w", encoding="utf-8") for v in variants}
        done = 0
        for c in cands:
            if args.minutes and (time.time() - t0) / 60 > args.minutes:
                print("time budget reached, stopping", flush=True)
                stop = True
                break
            rec = recs[c["id"]]
            prefix_user = prefix_user_text(rec, args.char_budget)
            pre, _ = render(prefix_user, "")
            ids = tok(pre, add_special_tokens=False, return_tensors="pt")["input_ids"]
            if ids.shape[1] > args.max_prefix_tokens:       # keep instruction/definitions (head) and the end of the conversation (tail)
                keep = args.max_prefix_tokens
                ids = torch.cat([ids[:, : keep // 3], ids[:, -(keep - keep // 3):]], dim=1)
            L = ids.shape[1]
            cache = prefill(ids)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            for variant in variants:
                res = []
                for cd in c["cands"]:
                    exs = [] if variant == "plain" else cd.get("examples_rag" if variant == "rag" else "examples_rand", [])
                    js = []
                    passes = [True] if args.format == "yn" else [True, False]
                    for yes_first in passes:
                        _, post = render(prefix_user, question_text(cd["agent"], cd["mode"], exs, yes_first, args.format))
                        sid = tok(post, add_special_tokens=False, return_tensors="pt")["input_ids"]
                        lo = next_token_logodds(sid, cache)
                        crop_to(cache, L)
                        js.append(lo if yes_first else -lo)       # swapped pass: the FIRST letter means "No", so log-odds of Yes is the negative
                    res.append({"agent": cd["agent"], "mode": cd["mode"], "rank": cd["rank"], "J": sum(js) / len(js), "passes": js, "n_examples": len(exs)})
                fouts[variant].write(json.dumps({"id": c["id"], "variant": variant, "prefix_tokens": L, "scores": res}) + chr(10))
                fouts[variant].flush()
            done += 1
            if done % 10 == 0 or done == len(cands):
                print(f"[{split}] {done}/{len(cands)} records ({(time.time() - t0) / 60:.1f} min, last prefix {L} tokens)", flush=True)
            del cache
        for fo in fouts.values():
            fo.close()
        (out_dir / f"run_info_{'+'.join(variants)}_{split}{sfx}.json").write_text(json.dumps(
            {"model": args.model, "four_bit": args.four_bit, "variants": variants, "format": args.format, "cands": cname, "records": done,
             "minutes": (time.time() - t0) / 60, "char_budget": args.char_budget, "max_prefix_tokens": args.max_prefix_tokens}))
    print("done", flush=True)


if __name__ == "__main__":
    main()
