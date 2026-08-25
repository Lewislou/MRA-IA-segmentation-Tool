#!/usr/bin/env bash
set -eo pipefail

export MODEL_NAME=smdnet
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

: "${nnUNet_raw:?Set nnUNet_raw}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed}"
: "${nnUNet_results:?Set nnUNet_results}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
cd "${ROOT}"

DATASET_ID="${DATASET_ID:-504}"
CONFIG="${CONFIG:-3d_lowres}"
FOLD="${FOLD:-0}"

export SMDNET_SMOKE=1
export SMDNET_SMOKE_ITERS="${SMDNET_SMOKE_ITERS:-1}"
export SAM_CHUNK_SIZE="${SAM_CHUNK_SIZE:-1}"

python -m nnunetv2.run.run_training "${DATASET_ID}" "${CONFIG}" "${FOLD}"
