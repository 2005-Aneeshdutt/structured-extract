# Running on Kaggle (free T4)

Copy-paste cells, in order. Notebook settings: **Accelerator → GPU T4 x2**
(you only use one, but the T4 x2 option gets you the T4 rather than a P100),
**Internet → On**, **Persistence → Files only**.

> **Why merge and quantize here instead of locally.** Merging a LoRA into fp16
> base weights is a CPU/RAM operation that peaks near 7 GB. A WSL2 instance on a
> 16 GB machine gets ~7.9 GB by default, which is too close to the line to trust
> at the end of a five-day data run. Kaggle gives you 30 GB of RAM and builds
> llama.cpp cleanly, so the whole merge → GGUF → Q4_K_M chain runs here and you
> download a ~1 GB file instead of fighting a memory ceiling.

---

## Cell 1 — secrets

Add these under **Add-ons → Secrets** first, then attach them to the notebook:
`HF_TOKEN` (write scope), `WANDB_API_KEY`.

```python
from kaggle_secrets import UserSecretsClient
import os

secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
```

## Cell 2 — code and data

Two options. **Preferred:** upload `data/processed/` as a Kaggle Dataset (it is
only a few MB) and clone the code, so the dataset is versioned separately from
the code.

```python
!git clone --depth 1 https://github.com/2005-Aneeshdutt/structured-extract /kaggle/working/structured-extract
%cd /kaggle/working/structured-extract

# the processed dataset, attached as a Kaggle Dataset in the sidebar
!mkdir -p data/processed
!cp -r /kaggle/input/structured-extract-data/* data/processed/
!ls data/processed
```

## Cell 3 — install

```python
# Do NOT install torch. Kaggle's image ships a torch built against its exact CUDA
# runtime; replacing it from PyPI breaks the driver match and costs the session.
!pip -q install "trl>=0.9,<0.12" "transformers>=4.44,<5" peft bitsandbytes wandb accelerate
```

Then **restart the kernel** before training — pip cannot rebind modules already
imported in the session.

> **Unsloth is deliberately not installed.** Its wheels pin an exact
> torch/CUDA pair (`unsloth[cu121-torch230]` wants torch 2.3), and Kaggle's
> image has moved on — it shipped **torch 2.10.0+cu128** as of the last run, so
> that extra cannot resolve. Forcing it replaces torch and breaks the driver
> match, which costs the whole session. `train.py` detects the absence and falls
> back to TRL + PEFT + bitsandbytes: same adapter, same result, roughly 2x
> slower. Budget 2-4 h per rank rather than 1-2 h, and plan for one rank per
> session.

Both pins are load-bearing, and they constrain *different* things:

- `trl<0.12` — 0.12 renamed `tokenizer` to `processing_class`, and later
  versions dropped `DataCollatorForCompletionOnlyLM`, which is what masks loss
  to the assistant turn.
- `transformers<5` — TRL 0.9-0.11 forward `tokenizer=` down to
  `transformers.Trainer`, and **v5 renamed that parameter**. Kaggle's image
  ships transformers 5.x, so pinning TRL alone still type-errors inside
  `SFTTrainer.__init__`, twenty minutes in, after the model has loaded.

`assert_trl_api()` checks both in the first five seconds.

## Cell 4 — train one rank

```python
!python training/train.py \
    --config training/configs/rank16.json \
    --dataset-path data/processed/hf \
    --output-dir /kaggle/working/outputs \
    --hub-model-id <HF_USER>/qwen2.5-1.5b-jobs-extract-r16 \
    --wandb-project structured-extract
```

Watch the first 50 steps. Three things tell you it is healthy:

- `trainable ... params (X%)` — **measured 2.04% at r=16** (18,464,768 of
  907,081,216) with attention+MLP targets. The denominator is the 4-bit
  quantized parameter count PEFT reports, not the 1.5B fp16 count, which is why
  the percentage looks higher than the usual "<1% of the model" shorthand.
- `peak N GB` from the memory callback — under ~13 GB leaves headroom
- `GEN EVAL {'eval/schema_compliance': ...}` at step 50 — should already be
  well above the base model. If it is near zero at step 150, something is wrong
  with the chat template, not the hyperparameters.

## Cell 5 — the rank ablation

Three runs. Each is an independent session-safe unit, so if you run out of time
you still have usable results.

```python
for r in (8, 16, 32):
    !python training/train.py \
        --config training/configs/rank{r}.json \
        --dataset-path data/processed/hf \
        --output-dir /kaggle/working/outputs \
        --wandb-project structured-extract
```

If wall clock is tight, run rank 16 alone in one session and the other two in a
second session. Nothing about the ablation requires them to share a process.

## Cell 6 — merge + quantize (the reason we are here)

```python
!apt-get -qq install -y cmake > /dev/null
!bash quantize/merge_and_quantize.sh \
    /kaggle/working/outputs/qwen2.5-1.5b-r16-a32/adapter \
    /kaggle/working/models
!ls -lh /kaggle/working/models
```

Expect ~1.0 GB for the Q4_K_M file. The 3.1 GB f16 intermediate is deleted
automatically unless you set `KEEP_F16=1`.

## Cell 7 — push everything

```python
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
USER = "<HF_USER>"

api.create_repo(f"{USER}/qwen2.5-1.5b-jobs-extract-GGUF", exist_ok=True)
api.upload_folder(
    folder_path="/kaggle/working/models",
    repo_id=f"{USER}/qwen2.5-1.5b-jobs-extract-GGUF",
    allow_patterns=["*.gguf"],
)
```

Then download the GGUF locally for evaluation — or just let
`eval/run_eval.py --backend gguf` pull it from the Hub.

---

## Time budget

| Step | Estimate | Notes |
|---|---|---|
| install | 3–5 min | Unsloth wheel is the slow part |
| one training run | 1–2 h | ~750 optimizer steps; **estimated, not measured** |
| full ablation | 3–6 h | fits one 9 h session, but leaves little slack |
| merge + GGUF + quantize | 15–25 min | llama.cpp build dominates |

The 9-hour session limit is the real constraint. If you want the ablation and
the quantization in one session, run rank 16 first and quantize immediately —
that way a timeout costs you the ablation arms, not the shippable artifact.

## If the session dies mid-run

`save_steps=100` and `save_total_limit=2` mean there is a checkpoint on disk.
With **Persistence → Files only** enabled, `/kaggle/working` survives, and
`trainer.train(resume_from_checkpoint=True)` picks it up. Without persistence
enabled, it does not — set it before you start, not after.
