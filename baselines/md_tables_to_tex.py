#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 make_three_tables.py 產出的 Markdown 表格轉為 IEEE/booktabs LaTeX。

用法：
  python md_tables_to_tex.py 三张表.md 输出.tex
特性：
  * 自動辨識 `### 表 N ...` 標題、`>` 形式的註釋、`| ... |` 表格
  * 粗體 `**x**` → \\textbf{x}
  * `±` 保留，`—` → `--`，`%` → `\\%`，`_` → `\\_`
  * 產出可直接 \\input 的 fragment（含 table 環境）
"""
from __future__ import annotations

import re
import sys

TABLE_NAMES = {
    1: ("tab:main", "SDPR\\_Herb 上與基線方法的對比（20 seeds）"),
    2: ("tab:ablation", "消融實驗（SDPR\\_Herb，3 seeds）"),
    3: ("tab:hyper", "超參數敏感性（SDPR\\_Herb，3 seeds）"),
}


def esc(s: str) -> str:
    s = s.strip()
    s = s.replace("**", "\x00")          # 暫存粗體標記
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
    s = s.replace("_", r"\_")
    s = s.replace("—", "--").replace("–", "--")
    s = s.replace("±", r"$\pm$")
    parts = s.split("\x00")
    out = []
    for i, p in enumerate(parts):
        out.append(r"\textbf{%s}" % p if i % 2 == 1 else p)
    return "".join(out)


def split_row(line: str):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def convert(md_path: str, tex_path: str):
    lines = open(md_path, encoding="utf-8").read().splitlines()
    out = []
    out.append("% 由 md_tables_to_tex.py 自動生成；請配合 \\usepackage{booktabs}")
    i = 0
    n_tab = 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"###\s*表\s*(\d+)", ln)
        if m:
            n_tab = int(m.group(1))
            label, caption = TABLE_NAMES.get(n_tab, ("tab:%d" % n_tab, "Table %d" % n_tab))
            # 找緊接的表格
            j = i + 1
            notes = []
            while j < len(lines) and not lines[j].strip().startswith("|"):
                if lines[j].strip().startswith(">"):
                    notes.append(lines[j].strip().lstrip("> ").strip())
                j += 1
            if j >= len(lines):
                i += 1
                continue
            rows = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(split_row(lines[j]))
                j += 1
            if len(rows) < 2:
                i = j
                continue
            header = rows[0]
            body = [r for r in rows[1:] if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
            ncol = len(header)
            align = "l" + "c" * (ncol - 1)
            out.append("")
            out.append(r"\begin{table}[t]")
            out.append(r"\centering")
            out.append(r"\caption{%s}" % caption)
            out.append(r"\label{%s}" % label)
            out.append(r"\begin{tabular}{%s}" % align)
            out.append(r"\toprule")
            out.append(" & ".join(esc(c) for c in header) + r" \\")
            out.append(r"\midrule")
            for r in body:
                r = (r + [""] * ncol)[:ncol]
                out.append(" & ".join(esc(c) for c in r) + r" \\")
            out.append(r"\bottomrule")
            out.append(r"\end{tabular}")
            if notes:
                out.append(r"\par\smallskip")
                out.append(r"\begin{minipage}{0.95\linewidth}\footnotesize")
                out.append(" ".join(esc(n) for n in notes))
                out.append(r"\end{minipage}")
            out.append(r"\end{table}")
            i = j
            continue
        i += 1

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("[ok] %s (%d 個 table)" % (tex_path, n_tab))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
