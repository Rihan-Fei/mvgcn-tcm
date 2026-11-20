#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SDPR_Herb 最终对比表：本模型 hard_bce，其余基线 BCE。

只读 results/*.jsonl，不碰训练。缺失 seed 的模型照实标注 n。
用法: python make_table_hb_vs_bce.py [--seeds 20]
"""
import argparse, glob, json, os, statistics as st

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# (显示名, [候选文件名 tag 前缀...]，取 seed 数最多的那个)
BASE_ROWS = [
    ("Popularity",        ["v2_SDPR_Herb_Popularity_U_sdpr2"]),
    ("MF",                ["v2_SDPR_Herb_MF_U_sdpr2"]),
    ("MLP",               ["v2_SDPR_Herb_MLP_U_sdpr2"]),
    ("NGCF",              ["v2_SDPR_Herb_NGCF_U_sdpr2"]),
    # 旧 tag 的 config 指纹与当前 BASE_ARGS 不一致，优先取补跑的新 tag
    ("LightGCN",          ["v2_SDPR_Herb_LightGCN_U_sdpr2_lg20",
                           "v2_SDPR_Herb_LightGCN_U_sdpr2"]),
    ("GAT",               ["v2_SDPR_Herb_GAT_U_sdpr2"]),
    ("SMGCN",             ["v2_SDPR_Herb_SMGCN_U_sdpr2"]),
    ("MVGCNLite",         ["v2_SDPR_Herb_MVGCNLite_U_sdpr2"]),
    ("MVGCN",             ["v2_SDPR_Herb_MVGCN_U_sdpr2"]),
    ("MV-GCN (graph only)", ["v2_SDPR_Herb_SMGCN_MVANA_U_sdpr2"]),
]
OURS = ("MV-GCN (ours)", "v2_SDPR_Herb_SMGCN_MVANA_hb_c1600")
OURS_BCE = ("MV-GCN (ours, BCE)", "v2_SDPR_Herb_SMGCN_MVANA_U_sdpr2_drf")

METRICS = ("P@5", "R@20", "F1@20")


def load(prefix, limit=None):
    p = os.path.join(RES, prefix + "_runs.jsonl")
    if not os.path.exists(p):
        return []
    rs = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rs.append(json.loads(line))
        except Exception:
            pass
    rs.sort(key=lambda r: r.get("_seed", 0))
    if limit:
        rs = rs[:limit]
    return rs


def agg(rs):
    out = {}
    for m in METRICS:
        v = [r[m] for r in rs if m in r]
        out[m] = (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0) if v else (float("nan"), 0.0)
    be = [r.get("_best_epoch", 0) for r in rs]
    out["best_ep"] = (min(be), max(be)) if be else (0, 0)
    out["n"] = len(rs)
    return out


def fmt(t, bold=False):
    m, s = t
    s = "%.4f" % m if t[1] == 0.0 else "%.4f±%.4f" % (m, s)
    return "**%s**" % s if bold else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20, help="每个模型最多取前 N 个 seed")
    a = ap.parse_args()

    rows = []
    for name, prefixes in BASE_ROWS:
        best_rs = []
        for pre in prefixes:
            rs = load(pre, a.seeds)
            if len(rs) > len(best_rs):
                best_rs = rs
        if best_rs:
            rows.append((name, "BCE", agg(best_rs), False))
    for name, pre in (OURS, OURS_BCE):
        rs = load(pre, a.seeds)
        if rs:
            rows.append((name, "hard_bce" if pre == OURS[1] else "BCE", agg(rs), pre == OURS[1]))

    # 基线最强值（排除本模型的任何行，含其 BCE 参考行）
    base = [r for r in rows if not r[0].startswith("MV-GCN (ours")]
    best = {m: max(r[2][m][0] for r in base) for m in METRICS} if base else {}
    best_name = {}
    for m in METRICS:
        cand = [r[0] for r in base if r[2][m][0] == best[m]]
        best_name[m] = cand[0] if cand else "-"

    print("SDPR_Herb (360 症状 / 753 药物 / test 3,443)")
    print()
    print("| 模型 | 损失 | n | P@5 | R@20 | F1@20 | best_ep |")
    print("|---|---|---|---|---|---|---|")
    for name, loss, g, is_ours in rows:
        be = "%d–%d" % g["best_ep"] if g["best_ep"][1] > g["best_ep"][0] else str(g["best_ep"][0])
        print("| %s | %s | %d | %s | %s | %s | %s |" % (
            name, loss, g["n"], fmt(g["P@5"], is_ours), fmt(g["R@20"], is_ours),
            fmt(g["F1@20"], is_ours), be))
    print()
    ours = [r for r in rows if r[3]]
    if ours and best:
        g = ours[0][2]
        print("本模型 vs 最强基线：")
        for m in METRICS:
            print("  %-6s %+0.4f  (vs %s %0.4f)" % (m, g[m][0] - best[m], best_name[m], best[m]))
    incomplete = [r[0] for r in rows if r[2]["n"] < a.seeds]
    if incomplete:
        print("\n注意：seed 未满 %d 个 —— %s" % (a.seeds, "、".join(incomplete)))


if __name__ == "__main__":
    main()
