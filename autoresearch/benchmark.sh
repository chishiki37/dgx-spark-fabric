#!/bin/bash
# Standard benchmark for autoresearch — improved warmup
# Usage: benchmark.sh [label]
API="http://localhost:8000/v1/chat/completions"
MODEL="deepseek-v4-ablit"
LABEL="${1:-exp}"

# Quick health check
if ! curl -s --max-time 5 http://localhost:8000/v1/models | grep -q "deepseek"; then
  echo "RESULT: SERVER_DOWN"
  exit 1
fi

python3 << 'PY'
import requests, time, json, concurrent.futures

API = "http://localhost:8000/v1/chat/completions"
MODEL = "deepseek-v4-ablit"

def send(msg, max_tokens=300, thinking=False):
    payload = {
        "model": MODEL,
        "messages": [{"role":"user","content":msg}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "chat_template_kwargs": {"thinking": thinking}
    }
    r = requests.post(API, json=payload, timeout=120)
    d = r.json()
    content = d["choices"][0]["message"].get("content","") or ""
    ctok = d.get("usage",{}).get("completion_tokens",0)
    return content, ctok

# --- THOROUGH warmup (triggers all kernel paths) ---
# 1. Short prompt, short output (basic decode)
send("hello", 10)
# 2. Medium prompt, medium output (prefill + MoE routing)
send("Write a Python function that takes a list of dictionaries and returns a new list sorted by the 'name' key. Include error handling.", 200)
# 3. Concurrent warmup (batch scheduling paths)
with concurrent.futures.ThreadPoolExecutor(4) as pool:
    list(pool.map(lambda _: send("Say hello in 3 languages.", 50), range(4)))

# --- Single-stream decode (thinking=false for consistency) ---
t0 = time.time()
c, ctok = send("Write a detailed Python implementation of a binary search tree with insert, delete, and search methods. Include type hints and docstrings.", 500, thinking=False)
t1 = time.time()
decode_tps = ctok / (t1 - t0)

# --- C=4 concurrency (thinking=false for consistency) ---
def run_one(_):
    t0 = time.time()
    c, tok = send("Explain how the TCP three-way handshake works, step by step, in detail.", 300, thinking=False)
    return time.time() - t0, tok

t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(4) as pool:
    results4 = list(pool.map(run_one, range(4)))
wall_4 = max(r[0] for r in results4)
agg4 = sum(r[1] for r in results4) / wall_4

# --- C=8 concurrency ---
t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(8) as pool:
    results8 = list(pool.map(run_one, range(8)))
wall_8 = max(r[0] for r in results8)
agg8 = sum(r[1] for r in results8) / wall_8

# --- Correctness (thinking=true, normal mode) ---
c, _ = send("What is the capital of France? Answer in one word.", 100, thinking=True)
ok_fact = "paris" in c.lower()
c, _ = send("What is 17 * 23? Just the number.", 100, thinking=True)
ok_math = "391" in c
c, _ = send("Write a Python lambda to square a number.", 200, thinking=True)
ok_code = "lambda" in c.lower()
correct = sum([ok_fact, ok_math, ok_code])

print(f"RESULT|decode={decode_tps:.1f}|c4={agg4:.1f}|c8={agg8:.1f}|correct={correct}/3|fact={'✅' if ok_fact else '❌'}|math={'✅' if ok_math else '❌'}|code={'✅' if ok_code else '❌'}")
PY
