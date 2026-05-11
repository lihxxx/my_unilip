#!/usr/bin/env bash
#
# End-to-end driver: SFT-edit 3-way comparison
#   pass 1: 1B baseline   (no DTR)
#   pass 2: 1B DTR
#   pass 3: 3B DTR
#
# Stages:
#   capture  : visualize_edit_compare.py --capture_attn --side all
#              -> generated/{base,dtr,dtr2}/sft_edit/en/{key}.png
#                 attn_grids/{key}/daam_grids_{base,dtr,dtr2}.npz
#                 selected_keys.json (auto-dumped)
#                 keywords_auto.json (auto-extracted)
#   export   : export_attn_pngs.py for ALL keys in selected_keys.json
#              -> png_export/{key}/{input,edited_*,attn_*}.png
#   plot     : plot_sft_edit.py for ALL keys -> 3-row compare figures
#              -> compare_figs/{key}/edit_compare.{pdf,png}
#   all      : capture -> export -> plot
#
# Usage:
#   bash scripts/run_sft_3way.sh capture
#   bash scripts/run_sft_3way.sh export
#   bash scripts/run_sft_3way.sh plot
#   bash scripts/run_sft_3way.sh all
set -euo pipefail

STAGE="${1:-all}"

# ─────────────────── Paths ───────────────────
BASE_CKPT="${BASE_CKPT:-/mnt/tidal-alsh01/dataset/zeus/lihongxiang/models/UniLIP-1B}"
DTR_CKPT="${DTR_CKPT:-/mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip/results/unilip_intern_vl_1b_sft_alignment_distill05_D6_dynamic6/checkpoint-2385}"
DTR2_CKPT="${DTR2_CKPT:-/mnt/tidal-alsh01/dataset/zeus/lihongxiang/models/UniLIP-3B}"
OUT_ROOT="${OUT_ROOT:-results/vis_sft_edit_3way}"
SHARD_ROOT="${SHARD_ROOT:-/mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip/data/edit_sft}"
SFT_TAR_GLOB="${SFT_TAR_GLOB:-${SHARD_ROOT}/*.tar}"

# ─────────────────── Knobs ───────────────────
SFT_SAMPLE_N="${SFT_SAMPLE_N:-50}"
SFT_SAMPLE_SEED="${SFT_SAMPLE_SEED:-1}"
SFT_MAX_PROMPT_WORDS="${SFT_MAX_PROMPT_WORDS:-10}"
SFT_MAX_KEYWORDS="${SFT_MAX_KEYWORDS:-4}"
SEED="${SEED:-42}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.0}"
STEP_WINDOW="${STEP_WINDOW:-0.4,0.6}"
LAYERS="${LAYERS:-}"
TARGET_SIZE="${TARGET_SIZE:-448,448}"

# ─────────────────── Helpers ───────────────────
echo_header() {
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

run_capture() {
  echo_header "[STAGE: capture] 3-way SFT edit (base / dtr / dtr2) with cross-attn"
  echo "  Base   : ${BASE_CKPT}"
  echo "  DTR    : ${DTR_CKPT}"
  echo "  DTR2   : ${DTR2_CKPT}"
  echo "  Out    : ${OUT_ROOT}"
  echo "  Sample : N=${SFT_SAMPLE_N}, seed=${SFT_SAMPLE_SEED}, "\
       "max_prompt_words=${SFT_MAX_PROMPT_WORDS}"
  echo "  Guide  : ${GUIDANCE_SCALE}, step_window=${STEP_WINDOW}, layers=${LAYERS:-all}"
  mkdir -p "${OUT_ROOT}"

  ARGS=(
    --baseline_model_path "${BASE_CKPT}"
    --dtr_model_path      "${DTR_CKPT}"
    --dtr2_model_path     "${DTR2_CKPT}"
    --out_root            "${OUT_ROOT}"
    --source              sft
    --sft_tar_glob        "${SFT_TAR_GLOB}"
    --sft_sample_n        "${SFT_SAMPLE_N}"
    --sft_sample_seed     "${SFT_SAMPLE_SEED}"
    --sft_max_prompt_words "${SFT_MAX_PROMPT_WORDS}"
    --sft_max_keywords    "${SFT_MAX_KEYWORDS}"
    --seed                "${SEED}"
    --guidance_scale      "${GUIDANCE_SCALE}"
    --step_window         "${STEP_WINDOW}"
    --side                all
    --capture_attn
  )
  [[ -n "${LAYERS}" ]] && ARGS+=(--layers "${LAYERS}")

  python -u scripts/visualize_edit_compare.py "${ARGS[@]}"

  echo
  echo ">> Capture finished. Key receipts:"
  echo "   ${OUT_ROOT}/selected_keys.json"
  echo "   ${OUT_ROOT}/keywords_auto.json"
  echo "   ${OUT_ROOT}/generated/{base,dtr,dtr2}/sft_edit/en/<key>.png"
  echo "   ${OUT_ROOT}/attn_grids/<key>/daam_grids_{base,dtr,dtr2}.npz"
}

run_export() {
  echo_header "[STAGE: export] standalone PNGs for every key in selected_keys.json"
  local sel="${OUT_ROOT}/selected_keys.json"
  if [[ ! -f "${sel}" ]]; then
    echo "ERROR: ${sel} not found. Run the 'capture' stage first." >&2
    exit 1
  fi
  # Pull the comma-joined key list with python (avoid jq dependency).
  local KEYS
  KEYS=$(python3 -c "
import json
rows = json.load(open('${sel}'))
print(','.join(r['key'] for r in rows))
")
  if [[ -z "${KEYS}" ]]; then
    echo "ERROR: no keys parsed from ${sel}" >&2
    exit 1
  fi
  local n_keys
  n_keys=$(python3 -c "import json; print(len(json.load(open('${sel}'))))")
  echo "  Exporting ${n_keys} keys (target ${TARGET_SIZE})..."

  python -u scripts/export_attn_pngs.py \
    --out_root       "${OUT_ROOT}" \
    --keys           "${KEYS}" \
    --shard_root     "${SHARD_ROOT}" \
    --target_size    "${TARGET_SIZE}" \
    --tokenizer_path "${BASE_CKPT}"

  echo
  echo ">> Export finished. Per-key PNGs under: ${OUT_ROOT}/png_export/<key>/"
}

run_plot() {
  echo_header "[STAGE: plot] 3-row compare figures via plot_sft_edit.py"
  local sel="${OUT_ROOT}/selected_keys.json"
  if [[ ! -f "${sel}" ]]; then
    echo "ERROR: ${sel} not found. Run the 'capture' stage first." >&2
    exit 1
  fi
  python -u scripts/plot_sft_edit.py \
    --out_root       "${OUT_ROOT}" \
    --shard_root     "${SHARD_ROOT}" \
    --tokenizer_path "${BASE_CKPT}" \
    --baseline_label "Baseline (1B)" \
    --dtr_label      "DTR (1B)" \
    --dtr2_label     "DTR (3B)"
  echo
  echo ">> Plot finished. Per-key compare figures: ${OUT_ROOT}/compare_figs/<key>/edit_compare.{pdf,png}"
}

case "${STAGE}" in
  capture) run_capture ;;
  export)  run_export  ;;
  plot)    run_plot    ;;
  all)
    run_capture
    run_export
    run_plot
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    echo "Use one of: capture | export | plot | all" >&2
    exit 1
    ;;
esac

echo
echo "Stage '${STAGE}' done. Outputs under: ${OUT_ROOT}/"
