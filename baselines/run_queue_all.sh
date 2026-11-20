#!/usr/bin/env bash
# Queue driver: reads a queue file of "<tag> <n_seeds> <extra args>" lines and
# runs the configurations assigned to worker W out of N (round-robin), skipping
# any configuration whose result file already holds the requested number of seeds.
#
# Usage:  bash run_queue_all.sh <worker_id> <n_workers> [tag_regex_filter]
#   QFILE=queues/joint_grid.txt          bash run_queue_all.sh 0 1
#   QFILE=queues/ablation_univariate.txt bash run_queue_all.sh 0 1
#
# Environment:
#   PYTHON            python interpreter to use (default: python)
#   MVANA_DATA_ROOT   dataset root     (default: <repo>/Data)
#   MVANA_RESULT_DIR  result directory (default: <repo>/baselines/results)
#
# Concurrency: two workers are the safe limit on a single 8.5 GB laptop GPU;
# three or more start thrashing on memory.
set -u
cd "$(dirname "$0")" || exit 1

PY="${PYTHON:-python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MVANA_DATA_ROOT="${MVANA_DATA_ROOT:-$(cd .. && pwd)/Data}"
export MVANA_RESULT_DIR="${MVANA_RESULT_DIR:-$(pwd)/results}"

W="${1:-0}"
N="${2:-1}"
FILTER="${3:-.}"
QFILE="${QFILE:-queues/ablation_univariate.txt}"
LOG="${QLOG:-q_w${W}.log}"

# Full model configuration shared by every ablation and sensitivity cell.
BASE=(--dataset sdpr --model SMGCN_MVANA --layers 3 --sana_calib layer --sana_alpha 0.2
      --sz_c 2 --d_kv 64 --mask_rate 0.2 --mfm MFM --direct_shrink 1.0
      --direct_gate_init 0.01 --l2_target e0 --dim 64 --lr 1e-3 --reg 1e-4
      --batch 1024 --epochs 1600 --eval_every 50 --patience 10 --device cuda
      --split_seed 2021 --self_loop 1 --weighting invfreq_pos --loss hard_bce --topk 50
      --views ss,hh --seeds 3 --seed_start 0)

echo "=== worker $W/$N start $(date '+%F %T') ===" >> "$LOG"

idx=0
while read -r tag nseed extra; do
  case "$tag" in
    ''|\#*|//*) continue ;;
  esac
  if (( idx % N != W )); then idx=$((idx+1)); continue; fi
  if ! echo "$tag" | grep -qE "$FILTER"; then idx=$((idx+1)); continue; fi
  JSONL="results/v2_SDPR_Herb_SMGCN_MVANA_${tag}_runs.jsonl"
  have=0
  if [ -f "$JSONL" ]; then
    have=$(grep -o '"_seed": *[0-9]*' "$JSONL" 2>/dev/null | sort -u | wc -l)
  fi
  if (( have >= nseed )); then
    echo "[skip] $tag ($have/$nseed) $(date '+%T')" >> "$LOG"
    idx=$((idx+1)); continue
  fi
  echo "[start] $tag ($have/$nseed) $(date '+%F %T')" >> "$LOG"
  mkdir -p locks
  if ! mkdir "locks/$tag" 2>/dev/null; then
    echo "[locked] $tag held by another worker $(date '+%T')" >> "$LOG"
    idx=$((idx+1)); continue
  fi
  "$PY" run_baselines_v2.py "${BASE[@]}" --tag "$tag" $extra >> "$LOG" 2>&1
  echo "[end]   $tag exit=$? $(date '+%F %T')" >> "$LOG"
  rmdir "locks/$tag" 2>/dev/null
  idx=$((idx+1))
done < "$QFILE"

echo "=== worker $W/$N DONE $(date '+%F %T') ===" >> "$LOG"
