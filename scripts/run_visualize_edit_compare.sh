#!/usr/bin/env bash
# End-to-end driver for the rebuttal edit-comparison pipeline.
#
# Stages (run separately so you can pause between them):
#   generate : full GEdit-Bench EN, both ckpts, no cross-attn.
#   score    : Qwen2.5-VL-72B-AWQ -> SC/PQ/O for both ckpts + edited_object.
#               Produces scores/score_diff.csv (sorted by delta_O desc) and
#               scores/selected_topk.json.
#   capture  : re-generate the picked subset WITH cross-attention capture.
#               Reads either selected_topk.json (default) or a custom keys
#               JSON via $ONLY_KEYS.
#   plot     : render 2x3 Baseline-vs-DTR figures for the picked subset.
#   all      : generate -> score -> (you pick keys) -> capture -> plot.
#
# Usage:
#   bash scripts/run_visualize_edit_compare.sh generate
#   bash scripts/run_visualize_edit_compare.sh score
#   # Look at results/vis_edit_compare/scores/score_diff.csv, decide K or
#   # write a custom keys JSON, e.g. results/vis_edit_compare/scores/my_keys.json
#   ONLY_KEYS=results/vis_edit_compare/scores/my_keys.json \
#       bash scripts/run_visualize_edit_compare.sh capture
#   bash scripts/run_visualize_edit_compare.sh plot
set -euo pipefail

STAGE="${1:-all}"

# ─────────────────── Paths ───────────────────
BASE_CKPT="${BASE_CKPT:-/mnt/tidal-alsh01/dataset/zeus/lihongxiang/models/UniLIP-1B}"
DTR_CKPT="${DTR_CKPT:-/mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip/results/unilip_intern_vl_1b_sft_alignment_distill05_D6_dynamic6/checkpoint-2385}"
OUT_ROOT="${OUT_ROOT:-results/vis_edit_compare}"

# ─────────────────── Knobs ───────────────────
LANGUAGE="${LANGUAGE:-en}"
SEED="${SEED:-42}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.5}"
MAX_CASES="${MAX_CASES:-0}"          # 0 = full set
TOP_K="${TOP_K:-8}"                  # for selected_topk.json + plot defaults
STEP_WINDOW="${STEP_WINDOW:-0.4,0.6}"
LAYERS="${LAYERS:-}"                 # empty = all DiT layers

# Stage-3 inputs (with sensible defaults)
ONLY_KEYS="${ONLY_KEYS:-${OUT_ROOT}/scores/selected_topk.json}"
KEYWORDS_JSON="${KEYWORDS_JSON:-${OUT_ROOT}/scores/edited_objects.json}"
PLOT_KEYS="${PLOT_KEYS:-}"           # comma-separated; empty -> use selected_topk

# ─────────────────── Helpers ───────────────────
COMMON_GEN_ARGS=(
  --baseline_model_path "${BASE_CKPT}"
  --dtr_model_path      "${DTR_CKPT}"
  --out_root            "${OUT_ROOT}"
  --language            "${LANGUAGE}"
  --seed                "${SEED}"
  --guidance_scale      "${GUIDANCE_SCALE}"
  --step_window         "${STEP_WINDOW}"
)
[[ -n "${LAYERS}" ]] && COMMON_GEN_ARGS+=(--layers "${LAYERS}")
[[ "${MAX_CASES}" != "0" ]] && COMMON_GEN_ARGS+=(--max_cases "${MAX_CASES}")

run_generate() {
  echo "============================================================"
  echo "[STAGE: generate] full GEdit-Bench ${LANGUAGE}, both ckpts, NO capture"
  echo "============================================================"
  mkdir -p "${OUT_ROOT}"
  python -u scripts/visualize_edit_compare.py "${COMMON_GEN_ARGS[@]}"
}

run_score() {
  echo "============================================================"
  echo "[STAGE: score] Qwen2.5-VL-72B-AWQ over the generated images"
  echo "============================================================"
  python -u scripts/score_edit_compare.py \
    --out_root "${OUT_ROOT}" \
    --language "${LANGUAGE}" \
    --top_k    "${TOP_K}" \
    --qwen_seed 42 \
    --qwen_model "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
  echo
  echo ">> Score CSV: ${OUT_ROOT}/scores/score_diff.csv"
  echo ">> Top-K candidates: ${OUT_ROOT}/scores/selected_topk.json"
}

# Stage-3 takes a JSON file with EITHER:
#   - a plain JSON list  ["key1","key2",...]   (used by visualize_edit_compare.py)
#   - the selected_topk.json shape { "keys": [...] }
# We normalise to a plain list at runtime.
_normalise_keys_json() {
  local in="$1"
  local out="${OUT_ROOT}/scores/_capture_keys.json"
  python - <<PY
import json, sys
src = json.load(open("${in}", "r", encoding="utf-8"))
if isinstance(src, dict) and "keys" in src:
    keys = src["keys"]
elif isinstance(src, list):
    keys = src
else:
    raise SystemExit(f"Unrecognised keys file: {type(src)}")
json.dump(keys, open("${out}", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"Normalised {len(keys)} keys -> ${out}")
PY
  echo "${out}"
}

run_capture() {
  echo "============================================================"
  echo "[STAGE: capture] re-generate picked subset WITH cross-attn"
  echo "  ONLY_KEYS     = ${ONLY_KEYS}"
  echo "  KEYWORDS_JSON = ${KEYWORDS_JSON}"
  echo "============================================================"
  if [[ ! -f "${ONLY_KEYS}" ]]; then
    echo "ERROR: ONLY_KEYS file not found: ${ONLY_KEYS}" >&2
    exit 1
  fi
  if [[ ! -f "${KEYWORDS_JSON}" ]]; then
    echo "ERROR: KEYWORDS_JSON file not found: ${KEYWORDS_JSON}" >&2
    exit 1
  fi
  KEYS_NORM=$(_normalise_keys_json "${ONLY_KEYS}" | tail -n 1)
  python -u scripts/visualize_edit_compare.py \
    "${COMMON_GEN_ARGS[@]}" \
    --capture_attn \
    --only_keys     "${KEYS_NORM}" \
    --keywords_json "${KEYWORDS_JSON}" \
    --no_skip_existing
}

run_plot() {
  echo "============================================================"
  echo "[STAGE: plot] render 2x3 figures"
  echo "============================================================"
  EXTRA=()
  if [[ -n "${PLOT_KEYS}" ]]; then
    EXTRA+=(--keys "${PLOT_KEYS}")
  elif [[ -f "${OUT_ROOT}/scores/selected_topk.json" ]]; then
    EXTRA+=(--topk_json "${OUT_ROOT}/scores/selected_topk.json" --top_k "${TOP_K}")
  else
    EXTRA+=(--top_k "${TOP_K}")
  fi
  python -u scripts/plot_edit_compare.py \
    --out_root "${OUT_ROOT}" \
    "${EXTRA[@]}"
}

case "${STAGE}" in
  generate) run_generate ;;
  score)    run_score    ;;
  capture)  run_capture  ;;
  plot)     run_plot     ;;
  all)
    run_generate
    run_score
    echo
    echo ">>> [all] Pausing semantics: 'all' will continue straight into capture+plot"
    echo ">>> using the auto-selected top-${TOP_K}. Override by re-running stages"
    echo ">>> separately with a custom ONLY_KEYS."
    run_capture
    run_plot
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    echo "Use one of: generate | score | capture | plot | all" >&2
    exit 1
    ;;
esac

echo
echo "Stage '${STAGE}' done. Outputs under: ${OUT_ROOT}/"
