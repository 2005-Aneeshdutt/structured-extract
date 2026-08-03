#!/usr/bin/env bash
# Merge the LoRA adapter into the base model, convert to GGUF, quantize to Q4_K_M.
#
#   ./quantize/merge_and_quantize.sh outputs/qwen2.5-1.5b-r16-a32/adapter models/
#
# Runs on the local box (WSL Ubuntu). Merging needs the fp16 base model in *system
# RAM*, not VRAM -- it is a CPU tensor operation -- so a 4 GB GPU is irrelevant
# here. Budget ~8 GB of RAM and ~12 GB of disk.
#
# Why Q4_K_M specifically
# ----------------------
# Q4_K_M is the accepted quality/size knee for models in this class: ~4.8
# bits/weight, keeping attention.wv and feed_forward.w2 at Q6_K where quantization
# error hurts most. For Qwen2.5-1.5B that is ~1.0 GB, which fits the free
# HuggingFace Spaces CPU tier with room for a KV cache.
#   Q8_0   ~1.6 GB  near-lossless, but 60% larger for a difference we could not
#                   measure on this task
#   Q4_K_M ~1.0 GB  <- shipped
#   Q3_K_M ~0.8 GB  visible degradation on structured output; the model starts
#                   dropping closing braces, which is fatal for JSON
#
# The claim "quantization did not hurt" is NOT assumed here -- verify_quantized.py
# measures it on 50 held-out examples and reports the delta.

set -euo pipefail

ADAPTER="${1:?usage: merge_and_quantize.sh <adapter_dir> <out_dir> [base_model]}"
OUT_DIR="${2:-models}"
BASE_MODEL="${3:-Qwen/Qwen2.5-1.5B-Instruct}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
QUANT="${QUANT:-Q4_K_M}"

NAME="$(basename "$(dirname "$ADAPTER")")"
MERGED_DIR="$OUT_DIR/${NAME}-merged"
GGUF_F16="$OUT_DIR/${NAME}-f16.gguf"
GGUF_Q="$OUT_DIR/${NAME}-${QUANT}.gguf"

mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# 1. Merge adapter into base weights
# ---------------------------------------------------------------------------
# Merged in fp16, NOT into the 4-bit weights used at training time. Merging a
# LoRA into a quantized base is lossy in a way that compounds with the later GGUF
# quantization, and the two errors are not independent. Dequantize once, merge,
# then quantize once -- one lossy step, measurable in isolation.
echo "==> merging $ADAPTER into $BASE_MODEL (fp16, CPU)"
python - "$ADAPTER" "$BASE_MODEL" "$MERGED_DIR" <<'PY'
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter, base, out = sys.argv[1:4]
model = AutoModelForCausalLM.from_pretrained(
    base, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True
)
model = PeftModel.from_pretrained(model, adapter, torch_dtype=torch.float16)
model = model.merge_and_unload()
model.save_pretrained(out, safe_serialization=True)
# Load the tokenizer from the adapter dir: training may have added a pad token,
# and a tokenizer/weight mismatch produces garbage that looks like a bad
# fine-tune rather than a packaging bug.
AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
print(f"merged -> {out}")
PY

# ---------------------------------------------------------------------------
# 2. llama.cpp
# ---------------------------------------------------------------------------
if [ ! -d "$LLAMA_CPP_DIR" ]; then
  echo "==> cloning llama.cpp into $LLAMA_CPP_DIR"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
fi
if [ ! -x "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]; then
  echo "==> building llama.cpp (CPU only; we quantize on CPU and the Space is CPU)"
  cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
  cmake --build "$LLAMA_CPP_DIR/build" --config Release -j "$(nproc)" --target llama-quantize llama-cli
fi
python -m pip install -q -r "$LLAMA_CPP_DIR/requirements.txt"

# ---------------------------------------------------------------------------
# 3. HF -> GGUF f16 -> Q4_K_M
# ---------------------------------------------------------------------------
echo "==> converting to GGUF f16"
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MERGED_DIR" --outfile "$GGUF_F16" --outtype f16

echo "==> quantizing to $QUANT"
"$LLAMA_CPP_DIR/build/bin/llama-quantize" "$GGUF_F16" "$GGUF_Q" "$QUANT"

ls -lh "$GGUF_F16" "$GGUF_Q"

# The f16 GGUF is a 3.1 GB intermediate with no further use once the quantized
# file exists. Kept only if KEEP_F16=1, because it is the single biggest
# consumer of disk in this repo.
if [ "${KEEP_F16:-0}" != "1" ]; then
  rm -f "$GGUF_F16"
  echo "removed intermediate $GGUF_F16 (set KEEP_F16=1 to keep it)"
fi

cat <<EOF

Done. Next:
  1. verify quality did not regress:
       python -m quantize.verify_quantized --gguf $GGUF_Q --adapter $ADAPTER --n 50
  2. push to the Hub:
       huggingface-cli upload <user>/<repo> $GGUF_Q $(basename "$GGUF_Q")
EOF
