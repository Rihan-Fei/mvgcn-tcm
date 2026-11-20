#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""消融的逐 seed 配对检验：完整模型 vs 各消融配置（仅取共同 seed）。

输出：
  ablation_paired.md   每配置的逐 seed 差值、均值差、符号一致数与配对 t
用法：
  python ablation_paired.py [--out DIR]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PREFIX = "v2_SDPR_Herb_SMGCN_MVANA_"
FULL = "hb_c1600"

ABL = [
    ("- S-S 视图", "abl_noss"),
    ("- H-H 视图", "abl_nohh"),
    ("- 双视图残差", "abl_noviews"),
    ("- SANA", "abl_nosana"),
    ("- VLR", "abl_novlr"),
    ("- MFM", "abl_nomfm"),
    ("- 直连先验", "abl_nodirect"),
    ("先验可学习", "abl_nofreeze"),
]

METRICS = ["P@5", "R@20", "F1@20"]


def load(tag):
    p = os.path.join(RES, PREFIX + tag + "_runs.jsonl")
    if not os.path.exists(p):
        return {}
    out = {}
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        out[r["_seed"]] = r
    return out


def paired_t(d):
    n = len(d)
    if n < 2:
        return float("nan")
    m = st.mean(d)
    sd = st.stdev(d)
    if sd == 0:
        return float("inf")
    return m / (sd / math.sqrt(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    full = load(FULL)
    if not full:
        print("[error] 找不到完整模型结果 %s%s" % (PREFIX, FULL))
        return

    lines = []
    lines.append("# 消融逐 seed 配对检验（SDPR_Herb）\n")
    lines.append("基准：完整模型 `%s`（hard_bce, topk=50），仅取与各消融配置共有的 seed 做配对。\n" % FULL)

    rows = []
    for label, tag in ABL:
        cur = load(tag)
        seeds = sorted(set(full) & set(cur))
        if not seeds:
            rows.append((label, tag, 0, None))
            continue
        rec = {"label": label, "tag": tag, "seeds": seeds}
        for m in METRICS:
            d = [cur[s][m] - full[s][m] for s in seeds]
            rec[m] = (st.mean(d), paired_t(d), sum(1 for x in d if x > 0), d)
        rows.append(rec)

    # 逐 seed 明细
    lines.append("## 逐 seed P@5\n")
    hdr = "| 配置 | " + " | ".join("seed %d" % s for s in sorted(full)[:3]) + " |"
    lines.append(hdr)
    lines.append("|" + "---|" * (hdr.count("|") - 1))
    base = [full[s]["P@5"] for s in (0, 1, 2)]
    lines.append("| **完整模型** | " + " | ".join("%.4f" % v for v in base) + " |")
    for rec in rows:
        if isinstance(rec, tuple):
            continue
        cells = []
        for s in (0, 1, 2):
            if s in rec["seeds"]:
                cells.append("%.4f" % (full[s]["P@5"] + rec["P@5"][3][rec["seeds"].index(s)]))
            else:
                cells.append("—")
        lines.append("| %s | %s |" % (rec["label"], " | ".join(cells)))

    # 配对差
    lines.append("\n## 配对差值（消融 − 完整模型）\n")
    lines.append("| 配置 | n | ΔP@5 | 符号 | ΔR@20 | 符号 | ΔF1@20 | 符号 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for rec in rows:
        if isinstance(rec, tuple):
            lines.append("| %s | 0 | — | — | — | — | — | — |" % rec[0])
            continue
        n = len(rec["seeds"])
        cells = []
        for m in METRICS:
            mean, t, npos, _ = rec[m]
            arrow = "↑" if mean > 0 else "↓"
            cells.append("%+.4f %s (%d/%d正)" % (mean, arrow, npos, n))
        lines.append("| %s | %d | %s | %s | %s |" % (rec["label"], n, cells[0], cells[1], cells[2]))

    # 配对 t
    lines.append("\n## 配对 t（自由度 = n−1）\n")
    lines.append("| 配置 | n | t(P@5) | t(R@20) | t(F1@20) |")
    lines.append("|---|---|---|---|---|")
    for rec in rows:
        if isinstance(rec, tuple):
            continue
        cells = []
        for m in METRICS:
            t = rec[m][1]
            cells.append("—" if math.isnan(t) else ("inf" if math.isinf(t) else "%+.2f" % t))
        lines.append("| %s | %d | %s | %s | %s |" % (rec["label"], len(rec["seeds"]), cells[0], cells[1], cells[2]))

    out = os.path.join(args.out, "ablation_paired.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("[ok] %s" % out)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
