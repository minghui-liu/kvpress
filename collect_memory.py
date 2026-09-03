#!/usr/bin/env python3
"""Summarise tier-10 memory-profiling runs (reason/results_mem) into a
Table 15 ready form, plus the KV-cache profile.

Peak allocated memory is max(prefill peak, decoding peak) per sample, where the
two are measured separately: evaluate.py resets CUDA peak stats at the
prefill/decode boundary, so the decoding figure excludes the prefill transient.

Usage:  python3 collect_memory.py [results_dir]
"""
import os, sys, json, re, collections

RES = sys.argv[1] if len(sys.argv) > 1 else "/home/dixi/kvpress/reason/results_mem"
LABEL = {"full":"Full","h2o":"H2O","knorm":"KNorm","rkv":"R-KV","snapkv":"SnapKV-D",
         "streaming_llm":"StreamingLLM","scope":"SCOPE","rpc":"RPC",
         "pyramidkv":"PyramidKV","turboquant":"TurboQuant","zipcache":"ZipCache"}
ORDER = ["Full","H2O","KNorm","R-KV","SnapKV-D","StreamingLLM","SCOPE","RPC"]
BUDGETS = [128, 256, 384, 512]

if not os.path.isdir(RES):
    print(f"{RES} does not exist yet -- tier 10 has not been run."); sys.exit(0)

rows = {}
for f in sorted(os.listdir(RES)):
    if not f.endswith("_score.json"):
        continue
    s = json.load(open(os.path.join(RES, f)))
    if not s.get("measure_memory"):
        print(f"  WARNING measure_memory=false in {f}"); continue
    press = LABEL.get(s.get("press_name"), s.get("press_name"))
    bud = s.get("cache_budget")
    jl = os.path.join(RES, f[:-len("_score.json")] + ".jsonl")
    recs = []
    if os.path.exists(jl):
        for line in open(jl):
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    def mx(key):
        vals = [r.get(key) for r in recs if isinstance(r.get(key), (int, float))]
        return max(vals) if vals else None

    rows[(press, bud)] = {
        "n": len(recs),
        "peak_alloc":   s.get("max_memory_usage_gb"),
        "peak_prefill": s.get("max_prefill_memory_usage_gb", mx("prefill_memory_usage")),
        "peak_decode":  s.get("max_decoding_memory_usage_gb", mx("decoding_memory_usage")),
        "baseline":     s.get("avg_baseline_memory_usage_gb", mx("baseline_memory_usage")),
        "cache_prefill": mx("prefill_cache_memory"),
        "cache_peak":    mx("peak_cache_memory"),
        "cache_final":   mx("final_cache_memory"),
    }

if not rows:
    print(f"No scored memory runs in {RES} yet."); sys.exit(0)

def fmt(v, w=8, p=2):
    return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'--':>{w}}"

print(f"=== {RES}: {len(rows)} configs ===\n")
hdr = f"{'press':14s}{'bud':>5s}{'n':>4s}{'peak':>9s}{'prefill':>9s}{'decode':>9s}{'baseline':>10s}{'cache_pf':>10s}{'cache_pk':>10s}{'cache_fin':>10s}"
print(hdr); print("-" * len(hdr))
for press in ORDER + [p for p in {k[0] for k in rows} if p not in ORDER]:
    for b in BUDGETS:
        r = rows.get((press, b))
        if not r: continue
        print(f"{press:14s}{b:>5d}{r['n']:>4d}"
              f"{fmt(r['peak_alloc'],9)}{fmt(r['peak_prefill'],9)}{fmt(r['peak_decode'],9)}"
              f"{fmt(r['baseline'],10)}{fmt(r['cache_prefill'],10,4)}"
              f"{fmt(r['cache_peak'],10,4)}{fmt(r['cache_final'],10,4)}")

print("\n=== Table 15 body (peak allocated GB) ===")
full = [rows.get(("Full", b), {}).get("peak_alloc") for b in BUDGETS]
fv = [v for v in full if v is not None]
if fv:
    if max(fv) - min(fv) > 0.05:
        print(f"%% NOTE Full varies across budgets ({min(fv):.2f}-{max(fv):.2f}); it should not. Check GPU exclusivity.")
    print(f"    Full & \\multicolumn{{4}}{{c}}{{{max(fv):.2f}}} \\\\")
for press in ORDER[1:]:
    cells = [rows.get((press, b), {}).get("peak_alloc") for b in BUDGETS]
    if all(c is None for c in cells): continue
    print(f"    {press} & " + " & ".join(f"{c:.2f}" if c is not None else "--" for c in cells) + r" \\")
missing = [p for p in ORDER if not any((p, b) in rows for b in BUDGETS)]
if missing:
    print(f"\n%% no memory runs for: {', '.join(missing)}")
