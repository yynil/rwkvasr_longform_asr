#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python3"
CONFIG_YAML="${CONFIG_YAML:-${REPO_ROOT}/configs/generated/sensevoice_rwkv_bestrq_probe_from_stage6c_step1500.yaml}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

export CUDA_VISIBLE_DEVICES
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export RICH_FORCE_TERMINAL="${RICH_FORCE_TERMINAL:-1}"

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m rwkvasr.cli.train_best_rq --config-yaml "${CONFIG_YAML}" "$@"
