#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f .env.local ]]; then set -a; source .env.local; set +a; fi
export VTC_WEBUI="$PWD"
if [[ ! ${CUDA_VISIBLE_DEVICES+x} ]]; then
  gpu_ids=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | paste -sd, - || true)
  if [[ -n "$gpu_ids" ]]; then export CUDA_VISIBLE_DEVICES="$gpu_ids"; fi
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${VTC_ASSETS_ROOT:-../VTC-3D-MLLM}/runtime/t48_render_deps:${PYTHONPATH:-}"
exec "${VTC_PYTHON:-python}" -m uvicorn app.server:app --host "${VTC_HOST:-127.0.0.1}" --port "${VTC_PORT:-8100}"
