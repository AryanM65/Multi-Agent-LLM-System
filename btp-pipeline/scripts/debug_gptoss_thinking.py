"""debug_gptoss_thinking.py — isolated repro test for the empty-Retriever-sample bug.

Calls the Ollama backend directly (bypassing the full pipeline) with the
Retriever's actual prompt shape, and prints both `content` and `thinking`
field lengths/text across several calls. This confirms or refutes the
hypothesis in correct_project_context.md Section 8.1: that gpt-oss:20b-cloud's
default reasoning pass consumes the num_predict budget before `content` is
ever produced.

Usage:
    python scripts/debug_gptoss_thinking.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import ollama

from src.config import OLLAMA_MODEL, MAX_TOKENS, DEFAULT_TEMPERATURE

QUESTION = "Were Scott Derrickson and Ed Wood of the same nationality?"
CONTEXT = (
    "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer. "
    "Edward Davis Wood Jr. (October 10, 1924 - December 10, 1978) was an American filmmaker."
)
PROMPT = (
    "You are a Retriever agent. Given the question and candidate paragraphs, "
    "select and return only the sentences that are directly relevant to answering "
    "the question. Do not add any commentary or explanation — only return the "
    "relevant sentences.\n\n"
    f"Question: {QUESTION}\n"
    f"Paragraphs:\n{CONTEXT}\n\n"
    "Relevant sentences:"
)

N_CALLS = 5


def call_raw(num_predict: int, think: bool = None):
    kwargs = dict(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        options={"temperature": DEFAULT_TEMPERATURE, "num_predict": num_predict},
    )
    if think is not None:
        kwargs["think"] = think
    return ollama.chat(**kwargs)


def summarize(label, resp):
    msg = resp.get("message", {})
    content = msg.get("content", "")
    thinking = msg.get("thinking", None)
    print(f"  [{label}] content_len={len(content)!r}  thinking_len={len(thinking) if thinking else 0}")
    print(f"    content : {content[:150]!r}")
    if thinking:
        print(f"    thinking: {thinking[:150]!r}")


def main():
    print(f"Model: {OLLAMA_MODEL}  MAX_TOKENS(current)={MAX_TOKENS}\n")

    print(f"=== Test A: current production settings (num_predict={MAX_TOKENS}, think unset) ===")
    for i in range(N_CALLS):
        try:
            resp = call_raw(num_predict=MAX_TOKENS)
            summarize(f"call {i}", resp)
        except Exception as e:
            print(f"  [call {i}] ERROR: {e}")

    print(f"\n=== Test B: think=False (attempt to disable reasoning) ===")
    for i in range(N_CALLS):
        try:
            resp = call_raw(num_predict=MAX_TOKENS, think=False)
            summarize(f"call {i}", resp)
        except Exception as e:
            print(f"  [call {i}] ERROR (backend may not support 'think' kwarg): {e}")

    print(f"\n=== Test C: much larger budget (num_predict=1024), think unset ===")
    for i in range(N_CALLS):
        try:
            resp = call_raw(num_predict=1024)
            summarize(f"call {i}", resp)
        except Exception as e:
            print(f"  [call {i}] ERROR: {e}")


if __name__ == "__main__":
    main()
