"""Align the official SDPR Herb splits with the loader used by run_baselines_v2.py.

The loader (``load_dataset``) expects ``Data/<name>/{train.txt,test.txt}`` with
one record per line::

    <symptom_id> <symptom_id> ... \t <herb_id> <herb_id> ...

SDPR ships ``train_id.txt`` / ``valid_id.txt`` / ``test_id.txt`` in exactly that
format, so preparation is a rename. The official validation file is kept as
``valid.txt`` but is not used as a separate partition: the unified protocol
holds out 10% of ``train.txt`` internally (fixed split seed 2021), so that the
test partition stays identical to the one used by the SDPR paper.

Download first (from the repository released with Yue et al., JBHI 2025)::

    git clone --depth 1 https://github.com/ywjawmw/SDPR
    mkdir -p Data/SDPR_Herb
    cp SDPR/data/Herb/*_id.txt Data/SDPR_Herb/

Then run this script.
"""
import os
import shutil

BASE = os.environ.get(
    "MVANA_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data"),
)
BASE = os.path.join(BASE, "SDPR_Herb")

for src, dst in (("train_id.txt", "train.txt"),
                 ("valid_id.txt", "valid.txt"),
                 ("test_id.txt", "test.txt")):
    s = os.path.join(BASE, src)
    d = os.path.join(BASE, dst)
    if os.path.exists(s):
        shutil.copyfile(s, d)
        print(f"copied {src} -> {dst}  ({os.path.getsize(d):,} bytes)")
    else:
        print(f"MISSING {s}")

# Validate line counts and format of the two files the protocol actually reads.
for fn in ("train.txt", "test.txt"):
    p = os.path.join(BASE, fn)
    n = bad = 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                bad += 1
                continue
            try:
                syms = [int(x) for x in parts[0].split()]
                herbs = [int(x) for x in parts[1].split()]
            except ValueError:
                bad += 1
                continue
            if syms and herbs:
                n += 1
    print(f"{fn}: valid prescriptions={n} malformed lines={bad}")
