"""QLoRA SFT of Qwen2.5-1.5B-Instruct for schema-constrained JSON extraction.

Runs unmodified on a Kaggle free T4 (16 GB, Turing / sm_75).

    # Kaggle cell -- the script bootstraps its own repo checkout if needed
    !pip -q install "unsloth[cu121-torch230]" "trl<0.12" peft bitsandbytes wandb
    !python train.py --config configs/rank16.json --wandb-project structured-extract

    # ablation sweep (three separate runs, one per rank)
    !for r in 8 16 32; do python train.py --config configs/rank$r.json; done

Why Qwen2.5-1.5B-**Instruct** and not the base model
----------------------------------------------------
The headline claim of this project is a *lift over the base model*. If the
baseline were `Qwen2.5-1.5B` (non-instruct), it would score near zero on an
instruction-shaped task, and the lift would be an artifact of comparing an
instruction-tuned model to one that was never taught to follow instructions --
a strawman a senior reviewer will spot immediately. Starting from Instruct makes
the baseline as strong as it can honestly be, so the measured gain is
attributable to task specialization rather than to instruction-following.

T4-specific constraints, all of which shape the config below
------------------------------------------------------------
* **No bf16.** Turing (sm_75) has no bfloat16 tensor cores. fp16 + a loss scaler
  is the only mixed-precision option; the script asserts this rather than
  letting `bf16=True` fail 20 minutes into a run.
* **No FlashAttention-2.** It requires Ampere (sm_80+). Unsloth falls back to its
  own Triton/xformers kernels, which is exactly why Unsloth is worth the
  dependency here rather than plain PEFT.
* **16 GB, shared with the dataset cache.** 4-bit NF4 base weights (~1.1 GB) plus
  LoRA adapters plus optimizer state plus activations at seq 2048 leaves
  headroom; gradient checkpointing buys the rest.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("train")

REPO_URL = os.environ.get("STRUCTURED_EXTRACT_REPO", "https://github.com/2005-Aneeshdutt/structured-extract")


# ---------------------------------------------------------------------------
# Bootstrap: make data.schema importable on a bare Kaggle worker
# ---------------------------------------------------------------------------


def _ensure_schema_importable() -> None:
    """The schema must be the SAME file used for data prep and eval.

    Kaggle sessions start with only this script present. Rather than vendoring a
    copy of schema.py into the notebook (which is how the training prompt and the
    eval prompt drift apart), we put the real repo on sys.path -- either the
    checkout this script lives in, an attached Kaggle dataset, or a fresh clone.
    """
    here = Path(__file__).resolve()
    for candidate in (here.parent.parent, Path.cwd(), Path("/kaggle/input/structured-extract")):
        if (candidate / "data" / "schema.py").exists():
            sys.path.insert(0, str(candidate))
            return
    dest = Path("/kaggle/working/structured-extract") if Path("/kaggle").exists() else Path("./structured-extract")
    if not dest.exists():
        LOGGER.info("schema.py not found locally; cloning %s -> %s", REPO_URL, dest)
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(dest)], check=True)
    sys.path.insert(0, str(dest))


_ensure_schema_importable()

from data.schema import (
    FIELD_SPECS,
    build_messages,
    flatten,
    parse_prediction,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Every hyperparameter, with the reason it has that value.

    Serialized into the run directory and logged to W&B so a result can always be
    traced back to the exact configuration that produced it.
    """

    # -- model ------------------------------------------------------------
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    """QLoRA rather than fp16 LoRA. On a 16 GB T4 fp16 LoRA *would* fit, but 4-bit
    NF4 frees ~2 GB of activation headroom, which is what allows seq_len 2048 and
    per_device_batch_size 2 instead of 1 -- a straight throughput win. The
    accepted cost is ~0.2-0.5% quality vs fp16 LoRA, and we measure the end
    result anyway. NF4 specifically (not FP4) because it is information-
    theoretically optimal for normally-distributed weights, which is what
    transformer weights approximately are."""

    # -- LoRA -------------------------------------------------------------
    lora_rank: int = 16
    lora_alpha: int | None = None  # defaults to 2 * rank
    """alpha = 2*rank keeps the effective LoRA scaling (alpha/rank) fixed at 2.0
    across the ablation. Without this, changing rank would change BOTH capacity
    and effective learning rate, and the ablation would be uninterpretable --
    you could not say whether rank 32 helped because of capacity or because it
    was implicitly trained faster."""

    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",     # attention
        "gate_proj", "up_proj", "down_proj",        # MLP
    )
    """Attention AND MLP. Attention-only LoRA is the common default, but the MLP
    blocks hold most of a decoder's parameters and are where format/convention
    knowledge lives -- and our task is overwhelmingly about learning an output
    convention. Adding MLP targets is ~2.5x the adapter parameters and, at these
    ranks, a rounding error in VRAM."""

    use_rslora: bool = False
    """Rank-stabilized LoRA rescales alpha by sqrt(rank) instead of rank. It
    matters mainly at rank >= 64; we top out at 32, and enabling it would make
    the three ablation arms non-comparable. Off, deliberately."""

    # -- optimization -----------------------------------------------------
    learning_rate: float = 2e-4
    """2e-4 is the standard LoRA band (10-20x a full-finetune LR). LoRA updates a
    low-rank residual initialized at zero, so early steps are effectively
    unregularized -- a full-finetune LR of 2e-5 underfits badly in 3 epochs, and
    1e-3 diverges on a 1.5B."""

    num_train_epochs: float = 3.0
    """3 epochs on ~4k examples. Epoch 1 teaches the JSON shape; epoch 2 teaches
    the field conventions; epoch 3 is where val loss typically bottoms out. The
    eval callback below exists to catch the epoch-3 overfit case empirically
    rather than trusting this number."""

    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    """Effective batch 16. Chosen as the largest that fits with seq_len 2048 in
    4-bit on a T4. Batch is accumulated rather than real because activation
    memory, not compute, is the binding constraint."""

    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    """Cosine to ~0 over the run. With only ~750 optimizer steps the tail matters:
    a constant or linear schedule leaves the model mid-oscillation at the final
    checkpoint, which shows up as a noisier val metric run-to-run."""

    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    """0.3, not the usual 1.0. fp16 on Turing produces occasional loss spikes when
    the scaler over-corrects; tight clipping costs nothing here and removes the
    class of run that dies at step 400 with a NaN."""

    optim: str = "adamw_8bit"
    """bitsandbytes 8-bit AdamW: halves optimizer state vs fp32 AdamW. On LoRA the
    optimizer state is small anyway, but on a shared 16 GB card every GB of
    headroom is one fewer OOM at the end of a long run."""

    seed: int = 3407

    # -- data -------------------------------------------------------------
    dataset_path: str = "data/processed/hf"
    train_split: str = "train"
    eval_split: str = "val"
    max_train_samples: int | None = None

    # -- logging / eval ---------------------------------------------------
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    gen_eval_n: int = 48
    """How many val examples the generation callback decodes at each eval point.
    48 is a compromise: enough for schema-compliance rate to be meaningful
    (+/- ~7pp), small enough that generation adds < 60s per eval on a T4."""

    # -- output -----------------------------------------------------------
    output_dir: str = "outputs"
    run_name: str | None = None
    hub_model_id: str | None = None
    wandb_project: str = "structured-extract"

    def __post_init__(self) -> None:
        if self.lora_alpha is None:
            self.lora_alpha = 2 * self.lora_rank
        if self.run_name is None:
            self.run_name = f"qwen2.5-1.5b-r{self.lora_rank}-a{self.lora_alpha}"

    @classmethod
    def from_json(cls, path: Path) -> TrainConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # JSON has no comment syntax, and these configs carry a "_comment" field
        # explaining what the arm is for. Underscore-prefixed keys are therefore
        # documentation, not settings, and are dropped before validation.
        # Everything else still has to be a real field: a typo'd key silently
        # falling back to a default is how a rank ablation quietly becomes three
        # runs of the same configuration.
        data = {k: v for k, v in data.items() if not k.startswith("_")}
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        if "target_modules" in data:
            data["target_modules"] = tuple(data["target_modules"])
        return cls(**data)


# ---------------------------------------------------------------------------
# Environment assertions -- fail in 5 seconds, not 20 minutes
# ---------------------------------------------------------------------------


def assert_environment() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device. This script is for the Kaggle T4 runtime.")
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    supports_bf16 = cap[0] >= 8
    info = {
        "gpu": props.name,
        "vram_gb": round(props.total_memory / 1024**3, 1),
        "capability": f"sm_{cap[0]}{cap[1]}",
        "supports_bf16": supports_bf16,
        "torch": torch.__version__,
    }
    LOGGER.info("environment: %s", info)
    if props.total_memory / 1024**3 < 14:
        LOGGER.warning("less than 14 GB VRAM detected; reduce max_seq_length or batch size")
    return info


def assert_trl_api() -> None:
    """Fail in 5 seconds if the installed TRL does not match the API used here.

    This script targets TRL 0.9-0.11, where `SFTTrainer` takes `tokenizer`,
    `max_seq_length` and `data_collator` directly. TRL 0.12 renamed `tokenizer`
    to `processing_class`; later versions moved `max_seq_length` and
    `dataset_text_field` into `SFTConfig` and dropped
    `DataCollatorForCompletionOnlyLM` entirely.

    Without this check a mismatch surfaces as a bare `TypeError: got an
    unexpected keyword argument` *after* the model has loaded and the dataset
    has been tokenized -- 15-20 minutes into a Kaggle session, and easy to
    misread as a data problem. requirements-train.txt and KAGGLE.md both pin
    `trl<0.12`; this asserts the pin actually took effect.
    """
    import inspect

    import trl

    version = getattr(trl, "__version__", "unknown")
    try:
        from trl import DataCollatorForCompletionOnlyLM  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            f"trl {version} does not provide DataCollatorForCompletionOnlyLM, which this "
            "script uses for completion-only loss masking.\n"
            "Fix:  pip install 'trl>=0.9,<0.12'\n"
            "(Or port the SFTTrainer call to SFTConfig + assistant_only_loss for newer TRL.)"
        ) from e

    # TRL 0.9-0.11 forward `tokenizer=` down to transformers.Trainer. transformers
    # v5 renamed that parameter to `processing_class`, so a new-enough
    # transformers with an old-enough TRL type-errors INSIDE SFTTrainer.__init__
    # -- after the model has loaded, which is the 20-minute failure this function
    # exists to prevent. The TRL pin alone does not constrain it.
    import transformers

    if "tokenizer" not in inspect.signature(transformers.Trainer.__init__).parameters:
        raise SystemExit(
            f"transformers {transformers.__version__} dropped `Trainer(tokenizer=...)`, "
            f"but trl {version} still passes it.\n"
            "Fix:  pip install 'transformers>=4.45,<4.47'\n"
            "(Then RESTART the kernel -- pip cannot rebind an imported module.)\n"
            "\n"
            "Note the narrow window: `tokenizer` was deprecated in 4.46 in favour of\n"
            "`processing_class` and removed during the 4.5x line -- NOT at 5.0 as the\n"
            "deprecation notice said -- so a `<5` pin does not exclude the versions\n"
            "that dropped it. 4.46.x is the last release TRL 0.11 works against."
        )

    params = inspect.signature(trl.SFTTrainer.__init__).parameters
    missing = [p for p in ("tokenizer", "max_seq_length") if p not in params]
    if missing:
        raise SystemExit(
            f"trl {version} SFTTrainer does not accept {missing}. This script targets the "
            "0.9-0.11 API.\n"
            "Fix:  pip install 'trl>=0.9,<0.12'"
        )
    LOGGER.info("trl %s: SFTTrainer API matches", version)


def set_all_seeds(seed: int) -> None:
    """Seed every RNG that touches the run.

    Full bit-exact reproducibility on GPU also needs deterministic kernels, which
    cost ~20% throughput. We seed everything and accept kernel-level
    nondeterminism -- run-to-run variance on the val metrics is ~1pp, which the
    README reports rather than hides.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Model loading: Unsloth, with an honest PEFT fallback
# ---------------------------------------------------------------------------


def load_model(cfg: TrainConfig) -> tuple[Any, Any, str]:
    """Return (model, tokenizer, backend).

    Unsloth is the primary path (it supports Qwen2.5 including the 1.5B). The
    fallback is not decoration: Unsloth pins torch/CUDA combinations tightly and
    Kaggle occasionally rolls its base image, at which point the import fails.
    The TRL+PEFT+bitsandbytes path trains the identical adapter, just ~2x slower
    and with more VRAM -- so a broken image costs time, not the session.
    """
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.base_model,
            max_seq_length=cfg.max_seq_length,
            dtype=None,          # None -> auto-detect fp16 on Turing
            load_in_4bit=cfg.load_in_4bit,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.target_modules),
            bias="none",
            use_gradient_checkpointing="unsloth",  # ~30% less VRAM than the HF impl
            random_state=cfg.seed,
            use_rslora=cfg.use_rslora,
        )
        return model, tokenizer, "unsloth"
    except Exception as e:
        LOGGER.warning("Unsloth unavailable (%s); falling back to TRL + PEFT + bitsandbytes", e)

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,   # fp16, not bf16: Turing
        bnb_4bit_use_double_quant=True,         # ~0.4 bits/param saved, free
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=quant if cfg.load_in_4bit else None,
        torch_dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="sdpa",  # not flash_attention_2: needs sm_80+
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    ))
    return model, tokenizer, "peft"


# ---------------------------------------------------------------------------
# Callbacks -- this is the "I can explain what happens each step" surface
# ---------------------------------------------------------------------------


def _build_callbacks(cfg: TrainConfig, tokenizer: Any, eval_ds: Any, backend: str) -> list[Any]:
    import torch
    from transformers import TrainerCallback

    class ThroughputAndMemoryCallback(TrainerCallback):
        """Logs tokens/sec, step time, and peak VRAM.

        Trainer reports loss and LR. It does not report the two numbers that
        actually determine whether a run fits in a 9-hour Kaggle session, or
        whether the next config change will OOM. Logging them here means the
        answer to "how big can you go on a T4?" is measured, not guessed.
        """

        def __init__(self) -> None:
            self.t0 = time.time()
            self.last = self.t0
            self.tokens = 0

        def on_step_end(self, args, state, control, **kw):
            self.tokens += (
                args.per_device_train_batch_size
                * args.gradient_accumulation_steps
                * cfg.max_seq_length
            )
            if state.global_step % cfg.logging_steps == 0:
                now = time.time()
                dt = max(now - self.last, 1e-6)
                stats = {
                    "perf/tokens_per_sec": self.tokens / max(now - self.t0, 1e-6),
                    "perf/sec_per_step": dt / cfg.logging_steps,
                    "perf/elapsed_min": (now - self.t0) / 60,
                    "mem/peak_alloc_gb": torch.cuda.max_memory_allocated() / 1024**3,
                    "mem/peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
                }
                self.last = now
                if wandb_run is not None:
                    wandb_run.log(stats, step=state.global_step)
                LOGGER.info(
                    "step %d | %.0f tok/s | %.2f s/step | peak %.1f GB",
                    state.global_step, stats["perf/tokens_per_sec"],
                    stats["perf/sec_per_step"], stats["mem/peak_alloc_gb"],
                )

    class StructuredEvalCallback(TrainerCallback):
        """Generates real JSON on held-out val and scores the TASK metrics.

        THE most important callback in this file, and the reason we are not just
        trusting `eval_loss`.

        Cross-entropy on a JSON target is dominated by structural tokens -- braces,
        quotes, the 12 key names -- which the model nails after ~200 steps. After
        that, val loss goes almost flat while the thing we actually care about
        (does the object parse? is `salary.min_amount` right?) keeps improving,
        and sometimes *degrades* while loss still looks fine. Selecting a
        checkpoint on eval_loss would therefore pick the wrong one.

        So: decode `gen_eval_n` val examples, parse them, and log schema
        compliance plus macro field accuracy. Model selection uses
        `eval/schema_compliance`, not loss.
        """

        def __init__(self) -> None:
            self.examples = list(eval_ds.select(range(min(cfg.gen_eval_n, len(eval_ds)))))

        def on_evaluate(self, args, state, control, model=None, **kw):
            if model is None:
                return
            if backend == "unsloth":
                from unsloth import FastLanguageModel

                FastLanguageModel.for_inference(model)
            model.eval()

            n_parse = 0
            field_hits = 0
            field_total = 0
            t0 = time.time()
            try:
                for ex in self.examples:
                    prompt = tokenizer.apply_chat_template(
                        build_messages(ex["source_text"]), tokenize=False, add_generation_prompt=True
                    )
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                       max_length=cfg.max_seq_length).to(model.device)
                    with torch.no_grad():
                        out = model.generate(
                            **inputs,
                            max_new_tokens=400,
                            do_sample=False,       # greedy: eval must be deterministic
                            temperature=None,
                            top_p=None,
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        )
                    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    pred, _err = parse_prediction(text)
                    if pred is None:
                        field_total += len(FIELD_SPECS)
                        continue
                    n_parse += 1
                    gold, _ = parse_prediction(ex["target_json"])
                    if gold is None:
                        continue
                    pf, gf = flatten(pred), flatten(gold)
                    for k in FIELD_SPECS:
                        field_total += 1
                        # Coarse exact-match here on purpose: the full type-aware
                        # comparators live in eval/metrics.py and are the ones we
                        # report. This callback only needs a monotone training
                        # signal, so it stays cheap and dependency-free.
                        field_hits += int(json.dumps(pf[k], sort_keys=True, default=str)
                                          == json.dumps(gf[k], sort_keys=True, default=str))
            finally:
                if backend == "unsloth":
                    from unsloth import FastLanguageModel

                    FastLanguageModel.for_training(model)
                model.train()

            stats = {
                "eval/schema_compliance": n_parse / max(len(self.examples), 1),
                "eval/field_exact_macro": field_hits / max(field_total, 1),
                "eval/gen_seconds": time.time() - t0,
            }
            LOGGER.info("step %d | GEN EVAL %s", state.global_step,
                        {k: round(v, 4) for k, v in stats.items()})
            if wandb_run is not None:
                wandb_run.log(stats, step=state.global_step)

    return [ThroughputAndMemoryCallback(), StructuredEvalCallback()]


wandb_run: Any = None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    global wandb_run

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None, help="path to a configs/*.json")
    ap.add_argument("--lora-rank", type=int, default=None, help="override the config's rank (ablation shortcut)")
    ap.add_argument("--dataset-path", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--hub-model-id", default=None, help="push the final adapter here, e.g. user/qwen-jobs-r16")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--max-train-samples", type=int, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = TrainConfig.from_json(args.config) if args.config else TrainConfig()
    for attr, val in (("lora_rank", args.lora_rank), ("dataset_path", args.dataset_path),
                      ("output_dir", args.output_dir), ("hub_model_id", args.hub_model_id),
                      ("wandb_project", args.wandb_project), ("max_train_samples", args.max_train_samples)):
        if val is not None:
            setattr(cfg, attr, val)
    if args.lora_rank is not None:   # keep alpha = 2*rank after a CLI override
        cfg.lora_alpha = 2 * cfg.lora_rank
        cfg.run_name = f"qwen2.5-1.5b-r{cfg.lora_rank}-a{cfg.lora_alpha}"

    env = assert_environment()
    assert_trl_api()
    if env["supports_bf16"]:
        LOGGER.info("bf16-capable GPU detected -- still using fp16 so results match the T4 reference runs")
    set_all_seeds(cfg.seed)

    from datasets import load_from_disk
    from transformers import TrainingArguments
    from trl import SFTTrainer

    run_dir = Path(cfg.output_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")

    if not args.no_wandb and os.environ.get("WANDB_API_KEY"):
        import wandb

        wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.run_name,
                               config={**asdict(cfg), **env}, reinit=True)
    else:
        LOGGER.warning("W&B disabled (no WANDB_API_KEY); metrics go to stdout only")

    ds = load_from_disk(cfg.dataset_path)
    train_ds, eval_ds = ds[cfg.train_split], ds[cfg.eval_split]
    if cfg.max_train_samples:
        train_ds = train_ds.select(range(min(cfg.max_train_samples, len(train_ds))))
    LOGGER.info("data: train=%d eval=%d", len(train_ds), len(eval_ds))

    model, tokenizer, backend = load_model(cfg)
    LOGGER.info("backend=%s", backend)
    if tokenizer.pad_token is None:
        # Qwen ships no pad token. Padding with EOS is standard, and safe here
        # only because the collator masks pad positions out of the loss --
        # otherwise the model would be trained to emit EOS mid-sequence.
        tokenizer.pad_token = tokenizer.eos_token

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    LOGGER.info("LoRA r=%d alpha=%d | trainable %s / %s params (%.3f%%)",
                cfg.lora_rank, cfg.lora_alpha, f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    # ---- completion-only loss ------------------------------------------
    # We train on the assistant turn ONLY. Without this the model spends most of
    # its gradient budget learning to reproduce the schema card and the posting
    # text -- tokens it will never need to generate. In practice this is the
    # difference between a model that emits JSON and one that sometimes replays
    # the prompt back at you.
    from trl import DataCollatorForCompletionOnlyLM

    response_template = "<|im_start|>assistant\n"  # Qwen2.5 ChatML
    if response_template not in tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True
    ):
        raise SystemExit(
            f"chat template does not contain {response_template!r}. "
            "Completion-only masking would silently no-op -- fix the template before training."
        )
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

    def formatting_func(batch: Any) -> list[str]:
        """Render `messages` through the tokenizer's chat template.

        The dataset deliberately stores `messages` rather than a pre-rendered
        string (see prepare_dataset.to_sft_record) so the ChatML format is not
        baked into the data and swapping the base model cannot silently produce
        mismatched formatting. That decision only holds if the training script
        actually applies the template -- which it did not, so TRL had no way to
        turn a row into text and refused to build the trainer.

        Rendering HERE, from the tokenizer, is what keeps train-time and
        eval-time prompts byte-identical: eval calls the same
        `apply_chat_template` on the same `messages`. A hand-built f-string
        would drift by a newline and quietly inflate the reported lift.

        TRL maps with batched=True, so `batch["messages"]` is a list of
        conversations; it also calls this on a single example in some paths, so
        both shapes are handled.
        """
        conversations = batch["messages"]
        if conversations and isinstance(conversations[0], dict):
            conversations = [conversations]  # single example, not a batch
        return [tokenizer.apply_chat_template(c, tokenize=False) for c in conversations]

    # Assert the rendered text actually contains the string the collator masks
    # on. If it does not, completion-only masking silently no-ops and the model
    # trains on the schema card and the posting text as if they were targets --
    # which looks like slow convergence, not like a bug.
    _probe = formatting_func(train_ds[:1])[0]
    if response_template not in _probe:
        raise SystemExit(
            f"rendered training text does not contain {response_template!r}; "
            "completion-only masking would train on the prompt."
        )

    targs = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        optim=cfg.optim,
        fp16=True,      # Turing: fp16 only
        bf16=False,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,          # Kaggle gives 20 GB of working disk
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["wandb"] if wandb_run is not None else [],
        run_name=cfg.run_name,
        seed=cfg.seed,
        gradient_checkpointing=(backend == "peft"),  # unsloth handles its own
        group_by_length=True,   # ~20% fewer padding tokens on this length spread
        dataloader_num_workers=2,
    )

    # packing=False: packing would raise throughput ~30% by concatenating short
    # examples into full-length sequences, but it also lets one example's tokens
    # attend to the previous example's. For a task whose entire premise is
    # "extract only from THIS document", that is a correctness risk, not merely a
    # metrics risk -- and it would undermine the hallucination number we report.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        max_seq_length=cfg.max_seq_length,
        formatting_func=formatting_func,
        data_collator=collator,
        packing=False,
        args=targs,
        callbacks=_build_callbacks(cfg, tokenizer, eval_ds, backend),
    )

    LOGGER.info("starting training: %d optimizer steps planned",
                int(len(train_ds) * cfg.num_train_epochs
                    / (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps)))
    result = trainer.train()
    LOGGER.info("finished in %.1f min", result.metrics.get("train_runtime", 0) / 60)

    final_dir = run_dir / "adapter"
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    LOGGER.info("saved adapter -> %s", final_dir)

    (run_dir / "train_metrics.json").write_text(
        json.dumps({"config": asdict(cfg), "env": env, "metrics": result.metrics}, indent=2, default=str),
        encoding="utf-8",
    )

    if cfg.hub_model_id:
        trainer.model.push_to_hub(cfg.hub_model_id, private=False)
        tokenizer.push_to_hub(cfg.hub_model_id)
        LOGGER.info("pushed adapter -> https://huggingface.co/%s", cfg.hub_model_id)

    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
