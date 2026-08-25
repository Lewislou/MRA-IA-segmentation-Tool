#!/usr/bin/env bash
set -eo pipefail

export MODEL_NAME=smdnet
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

: "${nnUNet_raw:?Set nnUNet_raw}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed}"
: "${nnUNet_results:?Set nnUNet_results}"
: "${INPUT_FOLDER:?Set INPUT_FOLDER}"
: "${OUTPUT_FOLDER:?Set OUTPUT_FOLDER}"
: "${MODEL_FOLDER:?Set MODEL_FOLDER}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

nnUNetv2_predict_from_modelfolder \
  -i "${INPUT_FOLDER}" \
  -o "${OUTPUT_FOLDER}" \
  -m "${MODEL_FOLDER}" \
  -f ${FOLDS:-0 1 2 3 4} \
  -chk "${CHECKPOINT:-checkpoint_best.pth}"
