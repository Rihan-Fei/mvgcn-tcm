# Data

The corpus used in this work is **not** redistributed here. It is third-party data
released by the authors of SDPR, and their terms apply.

## SDPR Herb

* Paper: W. Yue, W. Ji, X. Wang, X. Ma, P. Wang, X. Wang. *SDPR: Prescription
  recommendation with syndrome differentiation in traditional Chinese medicine.*
  IEEE Journal of Biomedical and Health Informatics, 29(5):3736–3749, 2025.
  doi:10.1109/JBHI.2025.3525507
* Repository: <https://github.com/ywjawmw/SDPR> (code and the `data/Herb` files)

### Preparation

```bash
git clone --depth 1 https://github.com/ywjawmw/SDPR /tmp/SDPR
mkdir -p Data/SDPR_Herb
cp /tmp/SDPR/data/Herb/*_id.txt Data/SDPR_Herb/
python baselines/prepare_sdpr_data.py
```

`prepare_sdpr_data.py` renames `train_id.txt`, `valid_id.txt`, `test_id.txt` to
`train.txt`, `valid.txt`, `test.txt`, which is the layout `load_dataset()` in
`run_baselines_v2.py` reads, and prints the record count of each file.

Expected counts after preparation:

| file | records | role |
|---|---|---|
| `train.txt` | 21,755 | split internally into 19,580 train / 2,175 validation |
| `test.txt` | 3,443 | held-out test partition, identical to the SDPR paper |
| `valid.txt` | 1,162 | official validation split; **not** used as a separate partition |

Record format, one prescription per line:

```
<symptom_id> <symptom_id> ... <TAB> <herb_id> <herb_id> ...
```

## Using a different root

```bash
export MVANA_DATA_ROOT=/path/to/Data
export MVANA_RESULT_DIR=/path/to/results
```
