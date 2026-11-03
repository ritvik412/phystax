#!/usr/bin/env bash
# Serve Muse Glimmer 30B with SGLang, no Docker (Meta cookbook:
# meta-oss-cookbook/muse-glimmer/inference-server/sglang.md). Run inside tmux.
#
#   bash scripts/serve_glimmer_sglang.sh GPU_INDEX [MODEL_PATH]
#
# MODEL_PATH defaults to the 17 GB GGUF Q4_K_M file (text only, which is all the
# agent needs). For BF16 pass meta-models/Muse-Glimmer-30B (needs ~70 GB free).
# Requires SGLang installed from source in its own venv (see README), because the
# PyPI release predates Glimmer support. Logs and a GPU snapshot go to
# runs/provenance/ as evidence the open-weight model really ran.
set -euo pipefail
GPU="${1:?usage: serve_glimmer_sglang.sh GPU_INDEX [MODEL_PATH]}"
MODEL="${2:-${HF_HOME:-$HOME/.cache/huggingface}/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf}"
cd "$(dirname "$0")/.."
mkdir -p runs/provenance
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="runs/provenance/${STAMP}_sglang_server.log"
nvidia-smi > "runs/provenance/${STAMP}_nvidia_smi_before.txt" || true
echo "GPU ${GPU}; model ${MODEL}; log -> ${LOG}"
CUDA_VISIBLE_DEVICES="${GPU}" sglang serve \
  --model-path "${MODEL}" \
  --served-model-name muse-glimmer \
  --reasoning-parser muse \
  --tool-call-parser muse \
  --mem-fraction-static 0.85 \
  --host 127.0.0.1 --port 30000 2>&1 | tee "${LOG}"
