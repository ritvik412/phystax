#!/usr/bin/env bash
# Serve Muse Glimmer 30B on this machine's GPU with vLLM, following Meta's
# cookbook (meta-oss-cookbook/muse-glimmer/quickstart). Notes from that guide:
#   * a dedicated image is required; `pip install vllm` will not serve Glimmer
#   * budget ~72 GB VRAM (bf16 weights ~60 GB): one 80 GB H100 fits if it is free
#   * both --enable-auto-tool-choice and --tool-call-parser muse_glimmer are needed,
#     or tool calls come back as plain text; --reasoning-parser keeps reasoning out of content
# Usage:  bash scripts/serve_glimmer.sh [GPU_INDEX]     (default GPU 0)
# Only use a GPU you are permitted to use. Check `nvidia-smi` first.
set -euo pipefail
GPU="${1:-0}"
cd "$(dirname "$0")/.."
mkdir -p runs/provenance
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="runs/provenance/${STAMP}_vllm_server.log"
nvidia-smi > "runs/provenance/${STAMP}_nvidia_smi_before.txt" || true
echo "Serving on GPU ${GPU}; log -> ${LOG}"
docker pull vllm/vllm-openai:muse-glimmer
docker run --rm --gpus "device=${GPU}" --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-openai:muse-glimmer \
  meta-models/Muse-Glimmer-30B \
  --served-model-name muse-glimmer \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  --generation-config auto 2>&1 | tee "${LOG}"
