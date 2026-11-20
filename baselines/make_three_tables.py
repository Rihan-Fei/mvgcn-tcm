#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SDPR_Herb 三张表：主表 / 消融表 / 超参敏感性。

只读 results/*.jsonl，不碰训练。缺失的配置照实标注 —— 不会用占位数字。
用法:
    python make_three_tables.py              # 打印 markdown
    python make_three_tables.py --out x.md   # 写入文件
"""
import argparse, json, os, statistics as st

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
METRICS = ("P@5", "R@20", "F1@20")
MAIN_COLS = ("P@5", "P@10", "P@20", "R@10", "R@20", "F1@20")
PREFIX = "v2_SDPR_Herb_"

# ---------- 主表 ----------
BASE_ROWS = [
    ("Popularity",           ["Popularity_U_sdpr2"]),
    ("MF",                   ["MF_U_sdpr2"]),
    ("MLP",                  ["MLP_U_sdpr2"]),
    ("NGCF",                 ["NGCF_U_sdpr2"]),
    ("LightGCN",             ["LightGCN_U_sdpr2_lg20", "LightGCN_U_sdpr2"]),
    ("GAT",                  ["GAT_U_sdpr2"]),
    ("SMGCN",                ["SMGCN_U_sdpr2"]),
    ("MVGCNLite",            ["MVGCNLite_U_sdpr2"]),
    ("MVGCN",                ["MVGCN_U_sdpr2"]),
    ("MV-GCN (graph only)",  ["SMGCN_MVANA_U_sdpr2"]),
]
OURS = "SMGCN_MVANA_hb_c1600"
OURS_BCE = "SMGCN_MVANA_U_sdpr2_drf"

# ---------- 消融 ----------
FULL = "SMGCN_MVANA_hb_c1600"
ABL = [
    ("完整模型",                        FULL),
    ("− S-S 视图",                      "SMGCN_MVANA_abl_noss"),
    ("− H-H 视图",                      "SMGCN_MVANA_abl_nohh"),
    ("− 双视图残差 (S-S & H-H)",        "SMGCN_MVANA_abl_noviews"),
    ("− SANA（改全局层组合）",          "SMGCN_MVANA_abl_nosana"),
    ("− VLR",                           "SMGCN_MVANA_abl_novlr"),
    ("− MFM 掩码训练",                  "SMGCN_MVANA_abl_nomfm"),
    ("− 直连先验（纯图）",              "SMGCN_MVANA_abl_nodirect"),
    ("先验可学习（不冻结）",            "SMGCN_MVANA_abl_nofreeze"),
]

# ---------- 超参 ----------
HP_GROUPS = [
    ("通道数 s", "sz_c", [
        ("1", "SMGCN_MVANA_hp_c1"), ("2", FULL), ("3", "SMGCN_MVANA_hp_c3")]),
    ("VLR 维度 d_kv", "d_kv", [
        ("32", "SMGCN_MVANA_hp_v32"), ("64", FULL), ("128", "SMGCN_MVANA_hp_v128")]),
    ("嵌入维度 d", "dim", [
        ("32", "SMGCN_MVANA_hp_d32"), ("64", FULL), ("128", "SMGCN_MVANA_hp_d128")]),
    ("SANA 损失权重 α", "sana_alpha", [
        ("0.1", "SMGCN_MVANA_hp_a01"), ("0.2", FULL), ("0.5", "SMGCN_MVANA_hp_a05")]),
    ("难负样本数 K", "topk", [
        ("20", "SMGCN_MVANA_hp_k20"), ("50", FULL), ("100", "SMGCN_MVANA_hp_k100")]),
]
# 注：多通道 GCN 编码器的层数在移植时固定为 2（对齐 TF 原版 len([128,256])），
# `--layers` 对 SMGCN_MVANA 不生效，故不列为可扫描超参。实测 L=1/2/4 三组逐 seed 结果
# 完全一致（best_ep 亦相同），可佐证该参数对本模型是空操作。


def load(tag):
    p = os.path.join(RES, PREFIX + tag + "_runs.jsonl")
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
    # 多進程競寫可能產生同一 seed 的重複行，按 seed 去重（保留最後一條）
    dedup = {}
    for r in rs:
        dedup[r.get("_seed", 0)] = r
    return [dedup[k] for k in sorted(dedup)]


def load_best(tags):
    """多候选 tag，取 seed 数最多的那个。"""
    best = []
    for t in tags:
        rs = load(t)
        if len(rs) > len(best):
            best = rs
    return best


def restrict(rs, seeds):
    """只保留指定 seed 的行，用于与短跑配置配平。"""
    keep = [r for r in rs if r.get("_seed") in seeds]
    return keep


def agg(rs):
    if not rs:
        return None
    out = {}
    for m in set(METRICS) | set(MAIN_COLS):
        v = [r[m] for r in rs if m in r]
        out[m] = (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0)
    be = [r.get("_best_epoch", 0) for r in rs]
    out["best_ep"] = (min(be), max(be))
    out["n"] = len(rs)
    return out


def cell(t, bold=False):
    if t is None:
        return "—"
    s = "%.4f" % t[0] if t[1] == 0.0 else "%.4f±%.4f" % t
    return "**%s**" % s if bold else s


def ep(g, bold=False):
    lo, hi = g["best_ep"]
    s = "%d–%d" % (lo, hi) if hi > lo else str(lo)
    return "**%s**" % s if bold else s


def table_main():
    hdr = "| 模型 | 损失 | n | " + " | ".join(MAIN_COLS) + " | best_ep |"
    L = ["### 表 1  SDPR_Herb 主结果", "",
         hdr, "|---" * (len(MAIN_COLS) + 4) + "|"]
    rows = []
    for name, tags in BASE_ROWS:
        g = agg(load_best(tags))
        if g:
            rows.append((name, "BCE", g, False))
    g_ours = agg(load(OURS))
    g_bce = agg(load(OURS_BCE))
    if g_bce:
        rows.append(("MV-GCN（本模型，BCE 对照）", "BCE", g_bce, False))
    if g_ours:
        rows.append(("**MV-GCN（本模型）**", "**hard_bce**", g_ours, True))
    for name, loss, g, b in rows:
        cells = " | ".join(cell(g[m], b) for m in MAIN_COLS)
        L.append("| %s | %s | %d | %s | %s |" % (name, loss, g["n"], cells, ep(g, b)))
    # 最强基线：排除本模型的任何行（含其 BCE 对照行）
    base = [r for r in rows if not r[0].replace("*", "").startswith("MV-GCN（本模型")]
    if g_ours and base:
        best = {m: max(r[2][m][0] for r in base) for m in METRICS}
        nm = {m: [r[0] for r in base if r[2][m][0] == best[m]][0] for m in METRICS}
        L += ["", "本模型 − 最强基线："]
        for m in METRICS:
            L.append("- %s **%+0.4f**（最强基线 %s，%.4f）" % (m, g_ours[m][0] - best[m], nm[m], best[m]))
    miss = [r[0] for r in rows if r[2]["n"] < 20]
    if miss:
        L += ["", "> seed 未满 20：%s" % "、".join(miss)]
    return L


def table_abl():
    L = ["### 表 2  消融实验（SDPR_Herb，3 seeds，协议与完整模型一致）", "",
         "| 配置 | P@5 | ΔP@5 | R@20 | ΔR@20 | F1@20 | ΔF1@20 | best_ep |",
         "|---|---|---|---|---|---|---|---|"]
    # 与消融同样是 3 seeds：完整模型只取相同的 seed 子集，保证逐 seed 可比
    ab_seeds = set()
    for name, tag in ABL:
        if tag == FULL:
            continue
        rs = load(tag)
        if rs and not ab_seeds:
            ab_seeds = {r.get("_seed") for r in rs}
    full_all = load(FULL)
    full = agg(restrict(full_all, ab_seeds)) if ab_seeds else agg(full_all)
    have_all = True
    for name, tag in ABL:
        if tag == FULL:
            g = full
        else:
            g = agg(load(tag))
        if g is None:
            have_all = False
            L.append("| %s | — | — | — | — | — | — | — |" % name)
            continue
        if tag == FULL:
            L.append("| **%s** | %s | — | %s | — | %s | — | %s |" % (
                name, cell(g["P@5"], True), cell(g["R@20"], True), cell(g["F1@20"], True), ep(g)))
        else:
            d5 = g["P@5"][0] - full["P@5"][0]
            dr = g["R@20"][0] - full["R@20"][0]
            df = g["F1@20"][0] - full["F1@20"][0]
            L.append("| %s | %s | %+0.4f | %s | %+0.4f | %s | %+0.4f | %s |" % (
                name, cell(g["P@5"]), d5, cell(g["R@20"]), dr,
                cell(g["F1@20"]), df, ep(g)))
    if not have_all:
        L += ["", "> 有配置尚未跑完，对应行留空（—）。"]
    if full:
        L += ["", "> 完整模型行取与消融相同的 %d 个 seed（n=%d）以逐 seed 配平；"
                  "其 20 seeds 主表值见上表。" % (len(ab_seeds) or 0, full["n"])]
    return L


def table_hp():
    L = ["### 表 3  超参数敏感性（SDPR_Herb，3 seeds，其余超参与完整模型一致）", "",
         "| 参数 | 取值 | n | P@5 | ΔP@5 | R@20 | ΔR@20 | F1@20 | best_ep |",
         "|---|---|---|---|---|---|---|---|---|"]
    hp_seeds = set()
    for label, _key, vals in HP_GROUPS:
        for v, tag in vals:
            if tag == FULL:
                continue
            rs = load(tag)
            if rs and not hp_seeds:
                hp_seeds = {r.get("_seed") for r in rs}
    full = agg(restrict(load(FULL), hp_seeds)) if hp_seeds else agg(load(FULL))
    for label, _key, vals in HP_GROUPS:
        for v, tag in vals:
            b = (tag == FULL)
            g = full if b else agg(load(tag))
            if g is None:
                L.append("| %s | %s | — | — | — | — | — | — | — |" % (label, v))
                continue
            if b:
                L.append("| %s | **%s** | %d | %s | — | %s | — | %s | %s |" % (
                    label, v, g["n"], cell(g["P@5"], True), cell(g["R@20"], True),
                    cell(g["F1@20"], True), ep(g)))
            else:
                L.append("| %s | %s | %d | %s | %+0.4f | %s | %+0.4f | %s | %s |" % (
                    label, v, g["n"], cell(g["P@5"]), g["P@5"][0] - full["P@5"][0],
                    cell(g["R@20"]), g["R@20"][0] - full["R@20"][0],
                    cell(g["F1@20"]), ep(g)))
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    L = (table_main() + [""] + table_abl() + [""] + table_hp())
    txt = "\n".join(L)
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(txt + "\n")
        print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
