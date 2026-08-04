# Every command needed to go from `git clone` to a deployed demo, in order.
# Run `make help` for the list.

PY ?= python
DATA_DIR ?= data/processed
ADAPTER ?= outputs/qwen2.5-1.5b-r16-a32/adapter
GGUF ?= models/qwen2.5-1.5b-r16-a32-Q4_K_M.gguf
PRED ?= results/raw_predictions

.DEFAULT_GOAL := help
.PHONY: help install lint test smoke data label prepare eval-all compare report robustness quantize verify app clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install local dependencies (see README for the torch CUDA wheel)
	$(PY) -m pip install -r requirements.txt

lint:  ## ruff
	ruff check .

test:  ## Unit tests
	pytest tests/ -q

# 250 postings, not 120: prepare_dataset refuses to build splits with fewer than
# 100 training examples left after test+val, and a smoke test that trips that
# guard tests nothing.
smoke:  ## End-to-end data pipeline on 250 postings with the mock teacher (no API key)
	$(PY) -m data.generate_synthetic --n 250 --max-corpus-rows 4000 --teacher mock \
		--out data/interim/smoke_labeled.jsonl --cache data/interim/smoke_cache.jsonl
	$(PY) -m data.prepare_dataset --in data/interim/smoke_labeled.jsonl \
		--out data/processed_smoke --stats-out results/dataset_stats_smoke.md \
		--test-size 25 --val-size 25

# ---------------------------------------------------------------------------
# 1. Data — the long pole. TWO PHASES, in this order, one teacher throughout.
#
#   phase 1 (gold):  1,250 postings x 3 samples = 3,750 calls
#   phase 2 (bulk):  4,000 postings x 1 sample  = 4,000 calls  (minus overlap)
#                    ------------------------------------------
#                    ~7,750 calls @ 1,500/day  ≈ 5 days wall clock
#
# Order matters. Phase 1 runs first so phase 2 can `--exclude-from` it: the two
# phases draw from overlapping corpus slices, and without that flag the same
# postings get labeled twice, burning quota and putting duplicate ids into the
# splitter.
#
# One teacher for both phases, deliberately. Using a cheaper model for bulk
# labels would halve the wall clock but train the student toward one model's
# conventions while grading it against another's — a train/test label mismatch
# that is not worth two days.
#
# Both phases are cached and resumable: re-run the same target daily after the
# quota resets and it picks up exactly where it stopped, at zero cost for work
# already done.
# ---------------------------------------------------------------------------
# Teacher transport for both labeling phases. `gemini` (default) has the larger
# free tier; `openrouter` swaps the vendor without touching the protocol:
#     make label-gold TEACHER=openrouter TEACHER_ARGS="--requests-per-day 1000"
TEACHER ?= gemini
TEACHER_ARGS ?=

label-gold:  ## PHASE 1 — held-out labels, 3-sample self-consistency vote (run this first)
	$(PY) -m data.generate_synthetic --n 1250 --teacher $(TEACHER) --n-samples 3 \
		--out data/interim/labeled_gold.jsonl \
		--audit-out results/label_audit.md $(TEACHER_ARGS)

label-bulk:  ## PHASE 2 — training labels, single greedy pass, skipping phase-1 postings
	$(PY) -m data.generate_synthetic --n 5250 --teacher $(TEACHER) --n-samples 1 \
		--exclude-from data/interim/labeled_gold.jsonl \
		--out data/interim/labeled.jsonl \
		--audit-out results/label_audit_bulk.md $(TEACHER_ARGS)

label: label-gold label-bulk  ## both phases, in order

prepare:  ## Clean, split (voted labels → val/test), format for SFT, export HF + JSONL
	$(PY) -m data.prepare_dataset \
		--in data/interim/labeled.jsonl \
		--gold-from data/interim/labeled_gold.jsonl \
		--out $(DATA_DIR)

data: label prepare  ## label + prepare

# ---------------------------------------------------------------------------
# 2. Training happens on Kaggle -- see README. Nothing here runs it locally.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3. Evaluation. Each arm writes a prediction file; scoring is a separate,
#    cheap, pure step so the tables can be rebuilt without re-generating.
# ---------------------------------------------------------------------------
eval-all:  ## Run all four arms on the held-out test split
	$(PY) -m eval.run_eval --backend hf --split test --out $(PRED)/base_0shot.json
	$(PY) -m eval.run_eval --backend hf --few-shot 3 --split test --out $(PRED)/base_3shot.json
	$(PY) -m eval.run_eval --backend hf --adapter $(ADAPTER) --split test --out $(PRED)/finetuned_r16.json
	$(PY) -m eval.run_eval --backend gemini --split test --out $(PRED)/gemini.json

eval-ablation:  ## Score the three rank variants on the VALIDATION split
	$(PY) -m eval.run_eval --backend hf --adapter outputs/qwen2.5-1.5b-r8-a16/adapter  --split val --out $(PRED)/val_r8.json
	$(PY) -m eval.run_eval --backend hf --adapter outputs/qwen2.5-1.5b-r16-a32/adapter --split val --out $(PRED)/val_r16.json
	$(PY) -m eval.run_eval --backend hf --adapter outputs/qwen2.5-1.5b-r32-a64/adapter --split val --out $(PRED)/val_r32.json

robustness:  ## Perturbation suite, fine-tuned vs base
	$(PY) -m eval.robustness_test --backend hf --adapter $(ADAPTER) --out $(PRED)/robustness_finetuned.json
	$(PY) -m eval.robustness_test --backend hf --out $(PRED)/robustness_base.json

compare:  ## Build results/comparison_table.md
	$(PY) -m eval.compare_models \
		--run "Base 0-shot=$(PRED)/base_0shot.json" \
		--run "Base 3-shot=$(PRED)/base_3shot.json" \
		--run "LoRA r=16 (ours)=$(PRED)/finetuned_r16.json" \
		--run "Gemini 2.0 Flash=$(PRED)/gemini.json" \
		--baseline "Base 0-shot" --ceiling "Gemini 2.0 Flash" --ours "LoRA r=16 (ours)"

report:  ## Charts, ablation table, failure analysis
	$(PY) -m eval.generate_report \
		--run "Base 0-shot=$(PRED)/base_0shot.json" \
		--run "Base 3-shot=$(PRED)/base_3shot.json" \
		--run "LoRA r=16 (ours)=$(PRED)/finetuned_r16.json" \
		--run "Gemini 2.0 Flash=$(PRED)/gemini.json" \
		--ours "LoRA r=16 (ours)" --baseline "Base 0-shot" \
		--ablation "r=8=$(PRED)/val_r8.json" \
		--ablation "r=16=$(PRED)/val_r16.json" \
		--ablation "r=32=$(PRED)/val_r32.json" \
		--robustness $(PRED)/robustness_finetuned.json \
		--robustness-baseline $(PRED)/robustness_base.json

# ---------------------------------------------------------------------------
# 4. Ship
# ---------------------------------------------------------------------------
quantize:  ## Merge adapter -> GGUF -> Q4_K_M
	bash quantize/merge_and_quantize.sh $(ADAPTER) models/

verify:  ## Confirm quantization did not regress quality (validation split)
	$(PY) -m quantize.verify_quantized --gguf $(GGUF) --adapter $(ADAPTER) --n 50

app:  ## Run the Gradio demo locally against local GGUF files
	$(PY) app/app.py --local models/

clean:  ## Remove generated artifacts (keeps data/interim so labeling stays resumable)
	rm -rf data/processed data/processed_smoke results/raw_predictions results/charts outputs models
