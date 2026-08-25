#!/usr/bin/env bash
set -eo pipefail

export MODEL_NAME=smdnet
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

: "${nnUNet_raw:?Set nnUNet_raw}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed}"
: "${nnUNet_results:?Set nnUNet_results}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
cd "${ROOT}"

DATASET_ID="${DATASET_ID:-504}"
CONFIG="${CONFIG:-3d_lowres}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
mkdir -p "${LOG_DIR}"

echo "[$(date)] SMD-Net 5-fold | GPU=${CUDA_VISIBLE_DEVICES} | ${DATASET_ID} ${CONFIG}"

for FOLD in 0 1 2 3 4; do
  LOG_FILE="${LOG_DIR}/smdnet_${CONFIG}_fold${FOLD}.log"
  echo "[$(date)] fold ${FOLD} -> ${LOG_FILE}"
  python -m nnunetv2.run.run_training "${DATASET_ID}" "${CONFIG}" "${FOLD}" > "${LOG_FILE}" 2>&1
  echo "[$(date)] fold ${FOLD} done"
done

echo "[$(date)] finished"
