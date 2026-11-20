#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-extract all nine metrics (P/R/F1 at 5, 10, 20) for the eleven models of
Table 2 from the per-seed jsonl files, and print both a verbatim listing and a
LaTeX body so the numbers in the manuscript can be checked against the source.
"""
import glob, json, os, statistics as st

RES = os.environ.get(
    "MVANA_RESULT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
)
METRICS = [f"{m}@{k}" for k in (5, 10, 20) for m in ("P", "R", "F1")]
# order used for the table: P@5 P@10 P@20 R@5 R@10 R@20 F1@5 F1@10 F1@20
COLS = ["P@5", "P@10", "P@20", "R@5", "R@10", "R@20", "F1@5", "F1@10", "F1@20"]

# display name, candidate tags (most seeds wins), seeds cap
ROWS = [
    ("Popularity",   ["v2_SDPR_Herb_Popularity_U_sdpr2"], 20),
    ("MF",           ["v2_SDPR_Herb_MF_U_sdpr2"], 20),
    ("MLP",          ["v2_SDPR_Herb_MLP_U_sdpr2"], 20),
    ("NGCF",         ["v2_SDPR_Herb_NGCF_U_sdpr2"], 20),
    ("LightGCN",     ["v2_SDPR_Herb_LightGCN_U_sdpr2_lg20",
                      "v2_SDPR_Herb_LightGCN_U_sdpr2"], 20),
    ("SMGCN",        ["v2_SDPR_Herb_SMGCN_U_sdpr2"], 20),
    ("MVGCN-Lite",   ["v2_SDPR_Herb_MVGCNLite_U_sdpr2"], 20),
    ("MVGCN-SANA",   ["v2_SDPR_Herb_MVGCN_U_sdpr2"], 20),
    ("MVGCN-NoPrior",["v2_SDPR_Herb_SMGCN_MVANA_U_sdpr2",
                      "v2_SDPR_Herb_SMGCN_MVANA_hb_nodr_c1600"], 20),
    ("MVGCN",        ["v2_SDPR_Herb_SMGCN_MVANA_hb_c1600"], 20),
]


def load(prefix, limit):
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
    # de-duplicate identical seeds (concurrent writers produced repeated lines)
    seen, out = set(), []
    for r in sorted(rs, key=lambda r: r.get("_seed", 0)):
        s = r.get("_seed", 0)
        if s in seen:
            continue
        seen.add(s)
        out.append(r)
    return out[:limit]


def main():
    data = {}
    for name, prefixes, cap in ROWS:
        best = []
        for pre in prefixes:
            rs = load(pre, cap)
            if len(rs) > len(best):
                best = rs
        if not best:
            print("MISSING", name)
            continue
        g = {}
        for m in COLS:
            v = [r[m] for r in best if m in r]
            g[m] = (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0) if v else (float("nan"), 0.0)
        data[name] = (g, len(best))

    # ---- plain listing
    hdr = "%-15s %3s " % ("model", "n") + " ".join("%13s" % c for c in COLS)
    print(hdr)
    print("-" * len(hdr))
    for name, (g, n) in data.items():
        print("%-15s %3d " % (name, n) + " ".join(
            "%13s" % ("%.4f±%.4f" % g[c]) for c in COLS))

    # ---- best per column among all rows except ours
    print("\nbest baseline per column:")
    for c in COLS:
        cand = [(nm, g[c][0]) for nm, (g, n) in data.items() if nm != "MVGCN"]
        nm, v = max(cand, key=lambda t: t[1])
        ours = data["MVGCN"][0][c][0]
        print("  %-6s %-11s %.4f   ours %.4f   diff %+0.4f" % (c, nm, v, ours, ours - v))

    # ---- LaTeX body
    def cell(t, bold=False):
        s = "%.4f$\\pm$%.4f" % t
        return "\\textbf{%s}" % s if bold else s

    lines = []
    for name, (g, n) in data.items():
        ours = name == "MVGCN"
        nm = "\\textbf{MVGCN}" if ours else name
        if n < 20:
            nm += "$^{\\dagger}$"
        cells = []
        for c in COLS:
            v = g[c][0]
            b = ours and v == max(d[c][0] for d in (x[0] for x in data.values()))
            cells.append(cell(g[c], b))
        lines.append("%-22s & %s \\\\" % (nm, " & ".join(cells)))
    print("\n=== LaTeX body ===")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
