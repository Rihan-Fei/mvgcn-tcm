# -*- coding: utf-8 -*-
"""同 seed 配对 t 检验：比较两个模型的逐 seed 结果。

用法: python paired_compare.py MODEL_A_FILE MODEL_B_FILE [更多文件...]
      python paired_compare.py --quick
"""
import json
import os
import sys
import statistics

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return {r["_seed"]: r for r in rows}


def paired_t(a, b):
    """a, b: 同长度列表 -> (mean_diff, t, p, dz)"""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return statistics.mean(d), float("nan"), float("nan"), float("nan")
    md = statistics.mean(d)
    sd = statistics.stdev(d)
    if sd == 0:
        return md, float("nan"), float("nan"), float("nan")
    t = md / (sd / (n ** 0.5))
    if HAS_SCIPY:
        p = 2 * (1 - stats.t.cdf(abs(t), df=n - 1))
    else:
        p = float("nan")
    dz = md / sd
    return md, t, p, dz


def compare(name_a, ra, name_b, rb, metrics=("P@5", "R@20", "F1@20")):
    common = sorted(set(ra) & set(rb))
    if not common:
        print(f"{name_a} vs {name_b}: 无共同 seed")
        return
    print(f"\n### {name_a}  vs  {name_b}   (配对 seed n={len(common)})")
    print(f"{'指标':<8}{'A':>16}{'B':>16}{'差值':>12}{'p 值':>12}{'dz':>8}")
    for m in metrics:
        a = [ra[s][m] for s in common]
        b = [rb[s][m] for s in common]
        md, t, p, dz = paired_t(a, b)
        ma, mb = statistics.mean(a), statistics.mean(b)
        ps = f"{p:.2e}" if p == p else "n/a"
        flag = " *" if (p == p and p < 0.05) else ""
        print(f"{m:<8}{ma:>10.4f}±{statistics.stdev(a):<5.4f}{mb:>10.4f}±{statistics.stdev(b):<5.4f}"
              f"{md:>+12.4f}{ps:>12}{dz:>+8.2f}{flag}")


def main():
    if len(sys.argv) >= 3 and not sys.argv[1].startswith("--"):
        files = sys.argv[1:]
        data = [(os.path.basename(f).replace("v2_Set2Set_", "").replace("_runs.jsonl", ""),
                 load(os.path.join(RESULT_DIR, os.path.basename(f))))
                for f in files]
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                compare(data[i][0], data[i][1], data[j][0], data[j][1])
        return

    # 默认：已有结果的关键两两比较
    pairs = [
        ("v2_Set2Set_MVGCNLite_final_runs.jsonl", "v2_Set2Set_MF_final_runs.jsonl"),
        ("v2_Set2Set_MVGCNLite_final_runs.jsonl", "v2_Set2Set_LightGCN_final_runs.jsonl"),
        ("v2_Set2Set_NGCF_final_runs.jsonl", "v2_Set2Set_MF_final_runs.jsonl"),
        ("v2_Set2Set_MF_final_runs.jsonl", "v2_Set2Set_LightGCN_final_runs.jsonl"),
    ]
    for fa, fb in pairs:
        pa = os.path.join(RESULT_DIR, fa)
        pb = os.path.join(RESULT_DIR, fb)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print("skip (missing):", fa, fb)
            continue
        compare(fa.replace("v2_Set2Set_", "").replace("_runs.jsonl", ""), load(pa),
                fb.replace("v2_Set2Set_", "").replace("_runs.jsonl", ""), load(pb))


if __name__ == "__main__":
    if not HAS_SCIPY:
        print("[warn] scipy 不可用，p 值将为 n/a")
    main()
