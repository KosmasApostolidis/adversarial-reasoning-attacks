#!/usr/bin/env bash
# Download HF fp16 + Ollama Q4 weights for the active benchmark VLMs.

set -euo pipefail

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

: "${HF_HOME:=./.hf_cache}"
: "${OLLAMA_HOST:=http://127.0.0.1:11434}"
: "${HF_TOKEN:=}"

echo "[download_models] HF_HOME=${HF_HOME}"
echo "[download_models] OLLAMA_HOST=${OLLAMA_HOST}"
mkdir -p "${HF_HOME}"

# ---------- HF fp16 surrogates (Phase 0/1) ----------
HF_MODELS=(
  "Qwen/Qwen2.5-VL-7B-Instruct"
  "llava-hf/llava-v1.6-mistral-7b-hf"
)

for repo in "${HF_MODELS[@]}"; do
  echo "[download_models] HF snapshot: ${repo}"
  python -c "
import os, sys
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError
tok = os.environ.get('HF_TOKEN') or None
try:
    snapshot_download(repo_id='${repo}', cache_dir='${HF_HOME}', token=tok)
except GatedRepoError as e:
    sys.stderr.write(f'[download_models] GATED: ${repo} — accept license at https://huggingface.co/${repo} then retry.\n')
    sys.exit(1)
"
done

# ---------- Ollama Q4 twins ----------
if ! command -v ollama >/dev/null 2>&1; then
  echo "[download_models] WARN: 'ollama' CLI not on PATH — skipping Ollama pulls." >&2
else
  OLLAMA_MODELS=(
    "qwen2.5vl:7b-q4_K_M"
    "llava:7b-v1.6-mistral-q4_K_M"
  )
  for tag in "${OLLAMA_MODELS[@]}"; do
    echo "[download_models] ollama pull ${tag}"
    ollama pull "${tag}"
  done
fi

echo "[download_models] done."
