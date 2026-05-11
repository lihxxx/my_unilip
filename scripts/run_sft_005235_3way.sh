#!/usr/bin/env bash
#
# One-shot driver for the sft_005235 paper-figure pipeline:
#   1. capture  : add 3B (dtr2) products into the existing vis_sft_edit/ root
#                 (base/dtr 1B already there, only the 3B pass is launched).
#   2. export   : write standalone PNGs (input + edited_{base,dtr,dtr2} +
#                 attn_<kw>_{base,dtr,dtr2}) at full 448x448, plus crop10/
#                 and crop20/ versions (center-crop 10%/20% per side).
#   3. plot     : render the 3-row compare figure
#                 (compare_figs/sft_005235/edit_compare.{pdf,png}).
#
# Stages:
#   capture | export | plot | all (default)
#
# Usage:
#   bash scripts/run_sft_005235_3way.sh           # all
#   bash scripts/run_sft_005235_3way.sh export    # just the standalone PNGs
#
# All paths are configurable via env vars (see "Paths" below).
set -euo pipefail

STAGE="${1:-all}"

# ─────────────────── Fixed identity ───────────────────
KEY="sft_005235"

# ─────────────────── Paths ───────────────────
BASE_DIR="${BASE_DIR:-/mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip}"
OUT_ROOT="${OUT_ROOT:-${BASE_DIR}/results/vis_sft_edit}"
SHARD_ROOT="${SHARD_ROOT:-${BASE_DIR}/data/edit_sft}"

BASE_CKPT="${BASE_CKPT:-/mnt/tidal-alsh01/dataset/zeus/lihongxiang/models/UniLIP-1B}"
DTR_CKPT="${DTR_CKPT:-${BASE_DIR}/results/unilip_intern_vl_1b_sft_alignment_distill05_D6_dynamic6/checkpoint-2385}"
DTR2_CKPT="${DTR2_CKPT:-/mnt/tidal-alsh01/dataset/zeus/lihongxiang/models/UniLIP-3B}"

# ─────────────────── Knobs ───────────────────
SEED="${SEED:-42}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.0}"
STEP_WINDOW="${STEP_WINDOW:-0.4,0.6}"
TARGET_SIZE="${TARGET_SIZE:-448,448}"
CROP_PCTS="${CROP_PCTS:-10,20}"

cd "${BASE_DIR}"

# Scratch files for the single-key capture call.
TMP_DIR="${OUT_ROOT}/_tmp_${KEY}"
ONLY_KEYS_JSON="${TMP_DIR}/_only_${KEY}.json"
KW_JSON="${TMP_DIR}/_kws_${KEY}.json"

echo_header() {
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

# ─────────────────── Stage 1: capture (3B only) ───────────────────
run_capture() {
  echo_header "[STAGE: capture] add 3B (dtr2) products for ${KEY} into ${OUT_ROOT}"

  # 1a) Sanity: selected_keys.json must already contain ${KEY}.
  local SEL="${OUT_ROOT}/selected_keys.json"
  if [[ ! -f "${SEL}" ]]; then
    echo "ERROR: ${SEL} not found." >&2
    exit 1
  fi
  if ! python3 -c "import json,sys; sys.exit(0 if any(r['key']=='${KEY}' for r in json.load(open('${SEL}'))) else 1)"; then
    echo "ERROR: ${KEY} not in ${SEL}; cannot resolve shard/stem/keyword." >&2
    exit 1
  fi

  # 1b) Build the only_keys + keyword JSONs for this single key.
  mkdir -p "${TMP_DIR}"
  echo "[\"${KEY}\"]" > "${ONLY_KEYS_JSON}"
  python3 - <<PY
import json
sel = json.load(open("${SEL}"))
row = next(r for r in sel if r['key']=='${KEY}')
json.dump({'${KEY}': row.get('keyword','')},
          open("${KW_JSON}","w"),
          ensure_ascii=False, indent=2)
print(f"keyword: {row.get('keyword','')}")
PY

  # 1c) Back up selected_keys.json (the capture call would otherwise overwrite
  # it with a 1-element list because only_keys_mode reduces items to {KEY}).
  local BAK="${SEL}.bak_pre_${KEY}_3b"
  cp "${SEL}" "${BAK}"

  # 1d) Run capture for the dtr2 (3B) side only.
  # baseline/dtr paths are required by the script; we still pass the 1B paths
  # so it doesn't crash, but --side dtr2 means only the 3B pass is executed.
  python -u scripts/visualize_edit_compare.py \
      --baseline_model_path "${BASE_CKPT}" \
      --dtr_model_path      "${DTR_CKPT}" \
      --dtr2_model_path     "${DTR2_CKPT}" \
      --out_root            "${OUT_ROOT}" \
      --source              sft \
      --side                dtr2 \
      --capture_attn \
      --only_keys     "${ONLY_KEYS_JSON}" \
      --keywords_json "${KW_JSON}" \
      --no_skip_existing \
      --seed                "${SEED}" \
      --guidance_scale      "${GUIDANCE_SCALE}" \
      --step_window         "${STEP_WINDOW}"

  # 1e) Restore original 50-key selected_keys.json.
  mv "${BAK}" "${SEL}"

  # Verify outputs.
  local DTR2_IMG="${OUT_ROOT}/generated/dtr2/sft_edit/en/${KEY}.png"
  local DTR2_NPZ="${OUT_ROOT}/attn_grids/${KEY}/daam_grids_dtr2.npz"
  echo
  echo ">> Verifying 3B products:"
  for p in "${DTR2_IMG}" "${DTR2_NPZ}"; do
    if [[ -f "${p}" ]]; then echo "   OK  ${p}"; else echo "   MISS ${p}"; fi
  done

  rm -rf "${TMP_DIR}"
}

# ─────────────────── Stage 2: export PNGs (with crops) ───────────────────
run_export() {
  echo_header "[STAGE: export] standalone PNGs (3 ckpts x crops ${CROP_PCTS})"
  python -u scripts/export_attn_pngs.py \
      --out_root       "${OUT_ROOT}" \
      --keys           "${KEY}" \
      --shard_root     "${SHARD_ROOT}" \
      --target_size    "${TARGET_SIZE}" \
      --crop_pcts      "${CROP_PCTS}" \
      --tokenizer_path "${BASE_CKPT}"

  echo
  echo ">> PNG export tree:"
  if command -v tree >/dev/null 2>&1; then
    tree "${OUT_ROOT}/png_export/${KEY}" || true
  else
    find "${OUT_ROOT}/png_export/${KEY}" -type f | sort
  fi
}

# ─────────────────── Stage 3: compare figure (3 rows) ───────────────────
run_plot() {
  echo_header "[STAGE: plot] 3-row compare figure for ${KEY}"
  python -u scripts/plot_sft_edit.py \
      --out_root       "${OUT_ROOT}" \
      --keys           "${KEY}" \
      --shard_root     "${SHARD_ROOT}" \
      --tokenizer_path "${BASE_CKPT}" \
      --baseline_label "Baseline (1B)" \
      --dtr_label      "DTR (1B)" \
      --dtr2_label     "DTR (3B)"

  echo
  echo ">> Compare figure:"
  ls -lh "${OUT_ROOT}/compare_figs/${KEY}/"
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
echo "Stage '${STAGE}' done. Key: ${KEY}. Out_root: ${OUT_ROOT}"
