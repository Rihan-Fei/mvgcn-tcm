# -*- coding: utf-8 -*-
"""聯合超參分析：alpha(SANA 融合權重) x kappa(直連先驗收縮強度)

輸出
  1. fig_contour_P5.png / fig_contour_R20.png   等高線 + 標註
  2. fig_marginal.png                           邊際曲線（固定一個軸，看另一個軸）
  3. fig_parallel.png                           平行坐標圖（25 個單元 × 5 軸）
  4. joint_hp_summary.md                        魯棒性統計

用法: python plot_joint_hp.py [--seeds-max K]
"""
import argparse
import glob
import json
import os
import re
import statistics as st

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

RES = "results"
OUT = "figs_joint"
ALPHAS = [0.00, 0.10, 0.20, 0.50]
KAPPAS = [0.50, 1.00, 2.00, 4.00]
METRICS = [("P@5", "P@5"), ("R@20", "R@20"), ("F1@20", "F1@20")]
PAT = re.compile(r"SMGCN_MVANA_g2d_a(\d+)_k(\d+)_runs\.jsonl$")


def load_cells(seeds_max=None):
    """回傳 {(alpha, kappa): {metric: (mean, std, n)}}"""
    cells = {}
    for p in glob.glob(os.path.join(RES, "v2_SDPR_Herb_SMGCN_MVANA_g2d_*_runs.jsonl")):
        m = PAT.search(p.replace("\\", "/"))
        if not m:
            continue
        a = int(m.group(1)) / 100.0
        k = int(m.group(2)) / 100.0
        by_seed = {}
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                by_seed[r["_seed"]] = r
        seeds = sorted(by_seed)[:seeds_max] if seeds_max else sorted(by_seed)
        if not seeds:
            continue
        d = {}
        for key, _ in METRICS:
            v = [by_seed[s][key] for s in seeds]
            d[key] = (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0, len(v))
        d["_seeds"] = seeds
        cells[(round(a, 2), round(k, 2))] = d
    return cells


def matrix(cells, key):
    M = np.full((len(KAPPAS), len(ALPHAS)), np.nan)
    N = np.zeros_like(M, dtype=int)
    for i, k in enumerate(KAPPAS):
        for j, a in enumerate(ALPHAS):
            c = cells.get((round(a, 2), round(k, 2)))
            if c:
                M[i, j] = c[key][0]
                N[i, j] = c[key][2]
    return M, N


def _smooth(M):
    """把格點矩陣內插成平滑曲面；回傳 (gx, gy, Z)。僅內插，不外推。"""
    nK, nA = M.shape
    mask = np.isnan(M)
    if mask.all():
        return None, None, None
    if mask.any():
        from scipy.interpolate import griddata
        pts = np.array([(x, y) for y in range(nK) for x in range(nA) if not mask[y, x]])
        vals = np.array([M[y, x] for y in range(nK) for x in range(nA) if not mask[y, x]])
        gx, gy = np.meshgrid(np.linspace(0, nA - 1, 160), np.linspace(0, nK - 1, 160))
        Z = griddata(pts, vals, (gx, gy), method="linear")
        nan = np.isnan(Z)
        if nan.any():                      # 凸包外用最近鄰補，避免留白條
            Zn = griddata(pts, vals, (gx, gy), method="nearest")
            Z = np.where(nan, Zn, Z)
    else:
        from scipy.ndimage import zoom
        Z = zoom(M, 16, order=1)
        gx, gy = np.meshgrid(np.linspace(0, nA - 1, Z.shape[1]), np.linspace(0, nK - 1, Z.shape[0]))
    return gx, gy, Z


def draw_contour_pair(cells):
    """雙面板等高線（P@5 | R@20），供論文單圖使用。"""
    pairs = [("P@5", "P@5"), ("R@20", "R@20")]
    mats = [(k, matrix(cells, k)[0]) for k, _ in pairs]
    if any(np.all(np.isnan(M)) for _, M in mats):
        print("[skip] fig_contour_pair 無資料")
        return
    nA, nK = len(ALPHAS), len(KAPPAS)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), dpi=220)
    for ax, (key, M) in zip(axes, mats):
        gx, gy, Z = _smooth(M)
        vmin, vmax = np.nanmin(M), np.nanmax(M)
        if np.isclose(vmin, vmax):
            vmin, vmax = vmin - 1e-4, vmax + 1e-4
        pad = 0.25 * (vmax - vmin)
        levels = np.linspace(vmin - pad, vmax + pad, 13)
        fill = ax.contourf(gx, gy, Z, levels=levels, cmap="RdYlBu_r", alpha=0.92, extend="both")
        ax.contour(gx, gy, Z, levels=np.linspace(vmin, vmax, 5), colors="k",
                   linewidths=0.55, alpha=0.35)
        for i in range(nK):
            for j in range(nA):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, "%.4f" % M[i, j], ha="center", va="center", fontsize=7.4,
                            zorder=6, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.72))
        best = np.unravel_index(np.nanargmax(M), M.shape)
        ax.plot(best[1] + 0.20, best[0] - 0.20, marker="*", ms=18, mfc="#00c853", mec="k",
                mew=0.8, zorder=7)
        if 1.00 in KAPPAS and 0.20 in ALPHAS:
            ax.plot(ALPHAS.index(0.20) + 0.20, KAPPAS.index(1.00) - 0.20, marker="o", ms=12,
                    mfc="none", mec="k", mew=1.7, zorder=7)
        ax.set_xticks(range(nA))
        ax.set_xticklabels(["%.2f" % a for a in ALPHAS])
        ax.set_yticks(range(nK))
        ax.set_yticklabels(["%g" % k for k in KAPPAS])
        ax.set_xlabel(r"SANA fusion weight $\alpha$", fontsize=10)
        ax.set_ylabel(r"prior shrinkage $\beta$", fontsize=10)
        ax.set_xlim(-0.5, nA - 0.5)
        ax.set_ylim(-0.5, nK - 0.5)
        ax.set_title(key, fontsize=11)
        cb = fig.colorbar(fill, ax=ax, pad=0.02, fraction=0.045)
        cb.ax.tick_params(labelsize=7)
        cb.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
        cb.set_label(key, fontsize=8)
    fig.suptitle("Joint effect of $\\alpha$ and $\\beta$ on SDPR_Herb    "
                 "(star = best cell, circle = default $\\alpha$=0.20, $\\beta$=1.0)",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "fig_contour_pair." + ext))
    plt.close(fig)
    print("[ok] fig_contour_pair.png/.pdf")


def draw_contour(cells, key, fname, fmt="%.4f"):
    M, N = matrix(cells, key)
    if np.all(np.isnan(M)):
        print("[skip] %s 無資料" % fname)
        return
    nA, nK = len(ALPHAS), len(KAPPAS)
    cols_ok = [j for j in range(nA) if not np.isnan(M[:, j]).all()]
    rows_ok = [i for i in range(nK) if not np.isnan(M[i, :]).all()]
    if len(cols_ok) < 2 or len(rows_ok) < 2:
        print("[skip] %s 資料不足（需在 α 與 β 兩方向各至少兩列）" % fname)
        return
    best = np.unravel_index(np.nanargmax(M), M.shape)
    base = (KAPPAS.index(1.00), ALPHAS.index(0.20)) if (1.00 in KAPPAS and 0.20 in ALPHAS) else None

    # 以格點索引作座標，網格間用雙線性內插平滑（僅內插，不外推）
    xx, yy = np.meshgrid(np.arange(nA), np.arange(nK))
    mask = np.isnan(M)
    if mask.any():
        from scipy.interpolate import griddata
        pts = np.array([(x, y) for y in range(nK) for x in range(nA) if not mask[y, x]])
        vals = np.array([M[y, x] for y in range(nK) for x in range(nA) if not mask[y, x]])
        gx, gy = np.meshgrid(np.linspace(0, nA - 1, 120), np.linspace(0, nK - 1, 120))
        Z = griddata(pts, vals, (gx, gy), method="linear")
        nan = np.isnan(Z)
        if nan.any():                      # 凸包外的空白用最近鄰補，避免留白條
            Zn = griddata(pts, vals, (gx, gy), method="nearest")
            Z = np.where(nan, Zn, Z)
    else:
        from scipy.ndimage import zoom
        Z = zoom(M, 12, order=1)
        gx, gy = np.meshgrid(np.linspace(0, nA - 1, Z.shape[1]), np.linspace(0, nK - 1, Z.shape[0]))

    fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=200)
    vmin, vmax = np.nanmin(M), np.nanmax(M)
    if np.isclose(vmin, vmax):
        vmin, vmax = vmin - 1e-4, vmax + 1e-4
    levels = np.linspace(vmin, vmax, 13)
    fill = ax.contourf(gx, gy, Z, levels=levels, cmap="RdYlBu_r", alpha=0.95, extend="both")
    cs = ax.contour(gx, gy, Z, levels=np.linspace(vmin, vmax, 6)[1:-1],
                    colors="k", linewidths=0.7, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=6.5, fmt=fmt)
    cbar = fig.colorbar(fill, ax=ax, pad=0.02)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.set_label(key, fontsize=9)

    for i in range(nK):
        for j in range(nA):
            if not np.isnan(M[i, j]):
                ax.text(j, i, fmt % M[i, j], ha="center", va="center", fontsize=7.5,
                        zorder=6, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.72))

    ax.plot(best[1], best[0], marker="*", ms=16, mfc="#00c853", mec="black", mew=0.8,
            label="best %.4f" % M[best], zorder=5)
    if base:
        ax.plot(base[1], base[0], marker="o", ms=9, mfc="none", mec="black", mew=1.5,
                label="default (%.2f, %g)" % (ALPHAS[base[1]], KAPPAS[base[0]]), zorder=5)
    ax.set_xticks(range(nA))
    ax.set_xticklabels(["%.2f" % a for a in ALPHAS])
    ax.set_yticks(range(nK))
    ax.set_yticklabels([("%g" % k) for k in KAPPAS])
    ax.set_xlabel(r"SANA fusion weight  $\alpha$", fontsize=10)
    ax.set_ylabel(r"prior shrinkage  $\beta$", fontsize=10)
    ax.set_xlim(-0.5, nA - 0.5)
    ax.set_ylim(-0.5, nK - 0.5)
    ax.set_title("%s over joint hyper-parameters (SDPR_Herb)" % key, fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print("[ok] %s  best %.4f @ alpha=%.2f beta=%g" % (fname, M[best], ALPHAS[best[1]], KAPPAS[best[0]]))


def draw_marginal(cells):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=200)
    colors = cm.viridis(np.linspace(0.05, 0.9, len(KAPPAS)))
    for c, k in zip(colors, KAPPAS):
        y = [cells.get((a, k), {}).get("P@5", (np.nan,))[0] for a in ALPHAS]
        axes[0].plot(ALPHAS, y, "o-", color=c, ms=4, lw=1.3, label=r"$\beta$=%g" % k)
    axes[0].set_xlabel(r"$\alpha$ (SANA fusion weight)")
    axes[0].set_ylabel("P@5")
    axes[0].set_title(r"P@5 vs $\alpha$ at each $\beta$", fontsize=10)
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)

    colors = cm.plasma(np.linspace(0.05, 0.85, len(ALPHAS)))
    for c, a in zip(colors, ALPHAS):
        y = [cells.get((a, k), {}).get("P@5", (np.nan,))[0] for k in KAPPAS]
        axes[1].semilogx(KAPPAS, y, "s-", color=c, ms=4, lw=1.3, label=r"$\alpha$=%.2f" % a)
    axes[1].set_xlabel(r"$\beta$ (prior shrinkage, log scale)")
    axes[1].set_xticks(KAPPAS)
    axes[1].set_xticklabels(["%g" % k for k in KAPPAS])
    axes[1].set_xlim(min(KAPPAS) * 0.85, max(KAPPAS) * 1.18)
    axes[1].xaxis.set_minor_formatter(plt.NullFormatter())
    axes[1].xaxis.offsetText.set_visible(False)
    axes[1].set_ylabel("P@5")
    axes[1].set_title(r"P@5 vs $\beta$ at each $\alpha$", fontsize=10)
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_marginal.png"))
    plt.close(fig)
    print("[ok] fig_marginal.png")


def draw_parallel(cells):
    """平行坐標：α、β 用序位對齊（等距類別），三個指標用原值歸一化。"""
    keys = sorted(cells.items())
    if not keys:
        print("[skip] fig_parallel.png 無資料")
        return
    ai = {a: i / max(len(ALPHAS) - 1, 1) for i, a in enumerate(ALPHAS)}
    bi = {k: i / max(len(KAPPAS) - 1, 1) for i, k in enumerate(KAPPAS)}
    rows, cols = [], []
    for (a, k), d in keys:
        if a not in ai or k not in bi:
            continue
        rows.append([ai[a], bi[k], d["P@5"][0], d["R@20"][0], d["F1@20"][0]])
        cols.append(d["P@5"][0])
    if not rows:
        print("[skip] fig_parallel.png 無資料")
        return
    X = np.array(rows)
    Y = X.copy()
    for j in range(2, X.shape[1]):          # 指標軸按各自 min-max 歸一化
        lo, hi = X[:, j].min(), X[:, j].max()
        Y[:, j] = (X[:, j] - lo) / (hi - lo if hi > lo else 1.0)

    names = [r"$\alpha$", r"$\beta$", "P@5", "R@20", "F1@20"]
    fig, ax = plt.subplots(figsize=(8.8, 4.5), dpi=220)
    cmap = plt.get_cmap("RdYlBu_r")
    cmin, cmax = min(cols), max(cols)
    for i, z in enumerate(Y):
        t = (cols[i] - cmin) / max(cmax - cmin, 1e-12)
        ax.plot(range(len(names)), z, "-", color=cmap(t), lw=1.6, alpha=0.8, zorder=2)

    # 每個軸的刻度文字：前兩軸用真實超參值，後三軸用指標真實值
    tick_txt = {
        0: ["%.2f" % a for a in ALPHAS],
        1: ["%g" % k for k in KAPPAS],
        2: ["%.4f" % X[:, 2].min(), "%.4f" % X[:, 2].max()],
        3: ["%.4f" % X[:, 3].min(), "%.4f" % X[:, 3].max()],
        4: ["%.4f" % X[:, 4].min(), "%.4f" % X[:, 4].max()],
    }
    for j in range(len(names)):
        ax.axvline(j, color="0.35", lw=0.9, zorder=1)
        if j < 2:                            # 類別軸：每個取值一個刻度
            fr = np.linspace(0, 1, len(tick_txt[j]))
        else:
            fr = np.array([0.0, 1.0])
        for f, txt in zip(fr, tick_txt[j]):
            ax.plot([j - 0.02, j + 0.02], [f, f], color="0.3", lw=0.9, zorder=3)
            ax.text(j + 0.045, f, txt, ha="left", va="center", fontsize=7, color="0.15",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10.5)
    ax.set_yticks([])
    ax.set_ylim(-0.09, 1.09)
    ax.set_xlim(-0.32, len(names) - 0.42)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Parallel coordinates: %d joint settings of ($\\alpha$, $\\beta$), colour = P@5"
                 % len(Y), fontsize=10)
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(cmin, cmax))
    cb = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.03)
    cb.set_label("P@5", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_parallel.png"))
    fig.savefig(os.path.join(OUT, "fig_parallel.pdf"))
    plt.close(fig)
    print("[ok] fig_parallel.png  (%d 條折線)" % len(Y))


def write_summary(cells):
    M, N = matrix(cells, "P@5")
    ok = ~np.isnan(M)
    vals = M[ok]
    best = np.unravel_index(np.nanargmax(M), M.shape)
    L = ["# 聯合超參分析：α × β（SDPR_Herb）", "",
         "- 網格：α ∈ {%s} × β ∈ {%s}，共 %d 個單元，已完成 %d 個"
         % (", ".join("%.2f" % a for a in ALPHAS), ", ".join("%g" % k for k in KAPPAS),
            len(ALPHAS) * len(KAPPAS), int(ok.sum())), ""]
    if ok.sum():
        thr = vals.max() - 0.005
        near = int((vals >= thr).sum())
        L += ["## 魯棒性", "",
              "- 已完成單元 P@5：均值 %.4f，極差 %.4f，標準差 %.4f"
              % (vals.mean(), vals.max() - vals.min(), vals.std(ddof=1) if len(vals) > 1 else 0.0),
              "- 最優：P@5=%.4f @ (α=%.2f, β=%g)" % (M[best], ALPHAS[best[1]], KAPPAS[best[0]]),
              "- 落在最優值 0.005 以內的單元：%d/%d（%.0f%%）→ 說明存在一片高原而非單點尖峰"
              % (near, ok.sum(), 100.0 * near / ok.sum()), ""]
        # 每個 α 下的極差（跨 β）
        L += ["## 逐 α 的跨 β 穩定性（P@5 極差越小＝對 β 越不敏感）", "",
              "| α | 平均 P@5 | 跨 β 極差 |", "|---|---|---|"]
        for j, a in enumerate(ALPHAS):
            col = M[:, j][~np.isnan(M[:, j])]
            if len(col):
                L.append("| %.2f | %.4f | %.4f |" % (a, col.mean(), col.max() - col.min()))
        L += ["", "## 逐 β 的跨 α 穩定性", "", "| β | 平均 P@5 | 跨 α 極差 |", "|---|---|---|"]
        for i, k in enumerate(KAPPAS):
            row = M[i][~np.isnan(M[i])]
            if len(row):
                L.append("| %g | %.4f | %.4f |" % (k, row.mean(), row.max() - row.min()))
    L += ["", "## 資料完整度", "", "| β \\ α | " + " | ".join("%.2f" % a for a in ALPHAS) + " |",
          "|---|" + "---|" * len(ALPHAS)]
    for i, k in enumerate(KAPPAS):
        cells_row = []
        for j in range(len(ALPHAS)):
            if np.isnan(M[i, j]):
                cells_row.append("—")
            else:
                cells_row.append("%.4f (n=%d)" % (M[i, j], N[i, j]))
        L.append("| %g | %s |" % (k, " | ".join(cells_row)))
    open(os.path.join(OUT, "joint_hp_summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("[ok] joint_hp_summary.md")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-max", type=int, default=None)
    ap.add_argument("--res", default="results")
    ap.add_argument("--out", default="figs_joint")
    a = ap.parse_args()
    global RES, OUT
    RES, OUT = a.res, a.out
    os.makedirs(OUT, exist_ok=True)
    cells = load_cells(a.seeds_max)
    print("讀到 %d 個單元" % len(cells))
    if not cells:
        return
    draw_contour(cells, "P@5", "fig_contour_P5.png")
    draw_contour(cells, "R@20", "fig_contour_R20.png")
    draw_contour_pair(cells)
    draw_marginal(cells)
    draw_parallel(cells)
    write_summary(cells)


if __name__ == "__main__":
    main()
