"""Evaluation harness: run one model over a split and score it.

    # fine-tuned adapter, 4-bit, on the sacred test split
    python -m eval.run_eval --backend hf --adapter outputs/qwen2.5-1.5b-r16-a32/adapter \
        --split test --out results/raw_predictions/finetuned_r16.json

    # base model, zero-shot and 3-shot
    python -m eval.run_eval --backend hf --split test --out results/raw_predictions/base_0shot.json
    python -m eval.run_eval --backend hf --few-shot 3 --split test --out results/raw_predictions/base_3shot.json

    # ceiling reference
    python -m eval.run_eval --backend gemini --split test --out results/raw_predictions/gemini.json

    # the quantized artifact that actually ships
    python -m eval.run_eval --backend gguf --gguf models/qwen-jobs-r16-Q4_K_M.gguf \
        --split test --out results/raw_predictions/finetuned_q4.json

Runs on the author's RTX 2050 (4 GB). That constraint is why every local backend
is 4-bit or GGUF: a merged fp16 1.5B is ~3.1 GB of weights, which leaves no room
for a 2k-token KV cache plus activations. 4-bit NF4 weights are ~1.1 GB and leave
the card comfortable.

Fairness controls -- the part that makes the comparison mean something
----------------------------------------------------------------------
Every arm gets: the same system prompt, the same user template (both from
data/schema.py), the same truncation budget, greedy decoding, and the same
max_new_tokens. The ONLY differences between arms are the weights and, for the
3-shot arm, the exemplars. If any of those knobs differed per-arm, the reported
lift would be partly prompt engineering, and the whole results table would be
unfalsifiable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from data.schema import (
    MAX_SOURCE_CHARS,
    SYSTEM_PROMPT,
    build_messages,
    build_user_prompt,
    gemini_response_schema,
    parse_prediction,
)
from eval.metrics import aggregate, bootstrap_ci, score_example

LOGGER = logging.getLogger("eval")

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 400
"""Enough for the longest valid object (a 15-skill posting serializes to ~330
tokens) plus headroom. Deliberately NOT generous: a base model that rambles
should be truncated and scored as non-compliant, because that is exactly what
would happen behind a production API with a token cap."""


class Backend(Protocol):
    name: str

    def generate(self, source_text: str, few_shot: list[tuple[str, str]] | None) -> str: ...


#: Backends may optionally expose `generate_batch`. `run()` uses it when present.
#:
#: This is not a micro-optimization. Measured on the author's RTX 2050, 4-bit
#: single-stream generation runs ~41 s/example: 500 test examples x 4 arms is
#: ~17 GPU-hours, and the robustness suite would add ~22 more. That is the
#: difference between an evaluation you run and one you keep postponing.
#: Batching amortizes the per-step kernel-launch and NF4 dequantization overhead,
#: which is what actually dominates at batch size 1.
BATCH_CAPABLE = "generate_batch"


# ---------------------------------------------------------------------------
# Local transformers backend (base + LoRA adapter, 4-bit)
# ---------------------------------------------------------------------------


class HFBackend:
    """transformers + bitsandbytes NF4, optionally with a PEFT adapter attached.

    Note we attach the adapter rather than merging it: merging into 4-bit weights
    is lossy in a way that is hard to reason about, and keeping the adapter
    separate means the base and fine-tuned arms share *identical* base weights,
    so any measured difference is the adapter. The merged model is evaluated
    separately, as GGUF, in quantize/verify_quantized.py.
    """

    def __init__(self, adapter: str | None = None, model_id: str = BASE_MODEL,
                 load_in_4bit: bool = True, batch_size: int = 8) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.name = f"hf:{Path(adapter).name if adapter else 'base'}"
        self.batch_size = batch_size
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ) if load_in_4bit else None

        self.tokenizer = AutoTokenizer.from_pretrained(adapter or model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # LEFT padding is mandatory for batched decoder-only generation. With
        # right padding, shorter prompts end in pad tokens and the model
        # continues from padding rather than from the prompt -- it produces
        # fluent garbage, and nothing in the output reveals the cause. This one
        # line is the difference between a correct batched eval and a silently
        # wrong one.
        self.tokenizer.padding_side = "left"

    def _render(self, source_text: str, few_shot: list[tuple[str, str]] | None) -> str:
        return self.tokenizer.apply_chat_template(
            build_messages(source_text, few_shot), tokenize=False, add_generation_prompt=True
        )

    def generate(self, source_text: str, few_shot: list[tuple[str, str]] | None = None) -> str:
        return self.generate_batch([source_text], few_shot)[0]

    def generate_batch(self, source_texts: list[str],
                       few_shot: list[tuple[str, str]] | None = None) -> list[str]:
        import torch

        prompts = [self._render(t, few_shot) for t in source_texts]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # greedy -- eval must be reproducible run to run
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # With left padding every sequence's completion starts at the same
        # offset, so a single slice is correct for the whole batch.
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# GGUF backend -- the artifact that actually ships
# ---------------------------------------------------------------------------


class GGUFBackend:
    """llama.cpp via llama-cpp-python. CPU-capable, which is what the Space runs.

    Evaluating the GGUF separately is not ceremony: quantization to Q4_K_M is a
    real perturbation of the weights, and "the merged fp16 model scores X" is not
    evidence that "the 1 GB file on HuggingFace scores X". The delta between this
    backend and HFBackend on the same split is reported in the README.
    """

    def __init__(self, gguf_path: str, n_ctx: int = 4096, n_gpu_layers: int = 0) -> None:
        from llama_cpp import Llama

        self.name = f"gguf:{Path(gguf_path).name}"
        self.llm = Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,  # 0 = pure CPU, matching HF Spaces free tier
            verbose=False,
            seed=0,
        )

    def generate(self, source_text: str, few_shot: list[tuple[str, str]] | None = None) -> str:
        resp = self.llm.create_chat_completion(
            messages=build_messages(source_text, few_shot),
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
        )
        return resp["choices"][0]["message"]["content"] or ""


# ---------------------------------------------------------------------------
# Gemini backend -- ceiling reference
# ---------------------------------------------------------------------------


class GeminiBackend:
    """Gemini 2.0 Flash, free tier.

    IMPORTANT CAVEAT, repeated in the README: Gemini also produced the gold
    labels. Its score here is therefore an upper bound biased in its own favor --
    it is closer to "self-consistency with the labeling protocol" than to
    accuracy. Two things keep the comparison honest:

      1. gold labels came from a *3-sample majority vote with grounding
         verification*; this arm is a single greedy pass, so the two are not the
         same procedure.
      2. results/label_audit.md scores the labels against LinkedIn's own
         non-LLM metadata, quantifying how good the labels are independently.

    The README states the gap-closing figure as conservative for this reason.
    """

    def __init__(self, model: str = "gemini-2.0-flash", constrained: bool = True) -> None:
        from google import genai

        from config import require_gemini_key

        self.client = genai.Client(api_key=require_gemini_key())
        self.model = model
        self.name = f"gemini:{model}" + ("" if constrained else ":unconstrained")
        self.constrained = constrained

    def generate(self, source_text: str, few_shot: list[tuple[str, str]] | None = None) -> str:
        from google.genai import types

        cfg: dict[str, Any] = {
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0.0,
            "max_output_tokens": 1024,
        }
        if self.constrained:
            # Constrained decoding is left ON for the ceiling arm on purpose: it
            # reflects how you would actually deploy Gemini for this task. A
            # reviewer may object that this hands Gemini a 100% schema-compliance
            # floor the local models do not get -- correct, and that is why
            # --unconstrained-gemini exists to produce the apples-to-apples row
            # as well. Both are reported.
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = gemini_response_schema()
        resp = self.client.models.generate_content(
            model=self.model,
            contents=build_user_prompt(source_text),
            config=types.GenerateContentConfig(**cfg),
        )
        return resp.text or ""


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def load_split(processed_dir: Path, split: str) -> list[dict[str, Any]]:
    path = processed_dir / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run data.prepare_dataset first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_fewshot(processed_dir: Path, k: int) -> list[tuple[str, str]]:
    """Exemplars come from a file written by prepare_dataset from TRAIN only.

    Loading them from disk rather than picking them here is deliberate: it makes
    it impossible for the few-shot arm to accidentally receive a test example,
    which would be silent contamination of the strongest baseline.
    """
    if k <= 0:
        return []
    path = processed_dir / "fewshot_exemplars.json"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run data.prepare_dataset first")
    ex = json.loads(path.read_text(encoding="utf-8"))
    return [(e["source_text"], e["target_json"]) for e in ex[:k]]


def build_backend(args: argparse.Namespace) -> Backend:
    if args.backend == "hf":
        return HFBackend(adapter=args.adapter, model_id=args.base_model,
                         load_in_4bit=not args.fp16, batch_size=args.batch_size)
    if args.backend == "gguf":
        if not args.gguf:
            raise SystemExit("--gguf PATH is required for the gguf backend")
        return GGUFBackend(args.gguf, n_gpu_layers=args.n_gpu_layers)
    if args.backend == "gemini":
        return GeminiBackend(args.gemini_model, constrained=not args.unconstrained_gemini)
    raise SystemExit(f"unknown backend {args.backend}")


def generate_completions(
    backend: Backend,
    examples: list[dict[str, Any]],
    few_shot: list[tuple[str, str]] | None = None,
    *,
    quiet: bool = False,
) -> list[tuple[str, float]]:
    """Generate one completion per example, batched and length-sorted.

    Returns [(completion, amortized_seconds), ...] in the SAME order as
    `examples`. Shared by run_eval and robustness_test -- the robustness suite
    makes ~1,000 generations per model, so running it unbatched would cost more
    wall clock than the entire headline evaluation.
    """
    batch_size = getattr(backend, "batch_size", 1) if hasattr(backend, BATCH_CAPABLE) else 1
    if batch_size > 1 and not quiet:
        LOGGER.info("batched generation, batch_size=%d â€” reported latency is amortized "
                    "per-example throughput, NOT single-request latency", batch_size)

    # Length-sorted batching. Padding is to the longest member of each batch, so a
    # mixed batch wastes compute on padding AND spikes KV-cache memory to the
    # worst case. Grouping similar lengths cuts both; results are restored to the
    # original order afterwards so scoring and the saved prediction file keep
    # dataset order.
    order = sorted(range(len(examples)), key=lambda i: len(examples[i]["source_text"]))
    outs_by_index: dict[int, tuple[str, float]] = {}
    cap = [batch_size]
    groups = _token_budget_batches(examples, order, batch_size)
    if not quiet:
        sizes = [len(g) for g in groups]
        LOGGER.info("token-budget batching: %d groups, sizes %d..%d (budget %d src chars)",
                    len(groups), min(sizes), max(sizes), BATCH_COST_BUDGET)

    done = 0
    for idxs in groups:
        chunk = [examples[i] for i in idxs]
        t0 = time.perf_counter()
        texts = _generate_chunk(backend, chunk, few_shot, cap)
        dt = (time.perf_counter() - t0) / max(len(chunk), 1)
        for i, text in zip(idxs, texts, strict=True):
            outs_by_index[i] = (text, dt)
        done += len(idxs)
        if not quiet:
            LOGGER.info("%d/%d | %.1fs/example | batch %d", done, len(order), dt, len(idxs))
    return [outs_by_index[i] for i in range(len(examples))]


#: Batch cost ceiling, in units of (source chars)^2.
#:
#: SQUARED, because the dominant allocation during prefill is the attention
#: score matrix -- batch x heads x seq x seq -- not the KV cache, which is only
#: linear in sequence length. Budgeting linearly gets the shape of the problem
#: wrong in both directions: it is far too strict on short documents (batches of
#: 1-2 where 16 would fit comfortably) and still too loose on the long tail.
#:
#: Calibrated from measurement on this 4 GB card rather than derived: batch 8 at
#: ~6,000 chars ran fine for hundreds of examples, so 8 x 6000^2 is a size known
#: to work. The observed failure -- batch 8 at 13,654 chars -- scores 1.5e9,
#: five times over, which is the run that consumed 23 minutes of CPU without
#: finishing a batch.
BATCH_COST_BUDGET = 8 * 6000 ** 2


def _token_budget_batches(
    examples: list[dict[str, Any]], order: list[int], max_size: int
) -> list[list[int]]:
    """Group length-sorted indices so each batch costs about the same memory.

    A fixed batch COUNT is the wrong unit. Padding runs to the longest member, so
    a batch of 8 short postings (~500 tokens each) and a batch of 8 long ones
    (~3,400 each) differ by ~7x in KV-cache footprint while looking identical in
    the code. Sorted ascending, that means the run gets steadily heavier and the
    last batches are the ones that do not fit.

    On Linux that surfaces as a clean OOM the caller can back off from. On
    Windows it does not: WDDM lets CUDA spill into system RAM rather than
    failing, so nothing raises. Measured here -- the 8 longest examples at batch
    size 8 consumed 23 minutes of CPU without completing a single batch, and
    later degraded to returning empty completions instantly. Every OOM handler
    in this file was waiting for an exception the platform never delivers.

    Budgeting by (longest x count) keeps peak memory roughly flat across the
    whole run: short groups come out large, long groups come out small, and the
    pathological case simply never forms. `max_size` still caps the batch so a
    corpus of very short documents does not build an enormous one.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    longest = 0
    for i in order:
        n = len(examples[i]["source_text"])
        candidate_longest = max(longest, n)
        # Cost model matches how padding actually works: every member is padded
        # up to the longest, so the batch costs longest x count.
        if current and (candidate_longest ** 2 * (len(current) + 1) > BATCH_COST_BUDGET
                        or len(current) >= max_size):
            groups.append(current)
            current, longest = [i], n
        else:
            current.append(i)
            longest = candidate_longest
    if current:
        groups.append(current)
    return groups


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"


def _free_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _generate_chunk(
    backend: Any, chunk: list[dict[str, Any]], few_shot: list[tuple[str, str]], cap: list[int]
) -> list[str]:
    """Generate for one chunk, halving the batch on CUDA OOM.

    Why OOM is handled differently from every other backend error
    -------------------------------------------------------------
    A timeout or an API error is scored as an empty completion, deliberately: a
    model whose service falls over is not thereby more accurate, and silently
    dropping its failures would flatter it.

    An OOM is not that. It is a property of the machine running the harness --
    a 4 GB card here -- and has nothing to do with the model under test.
    Recording it as an empty completion charges our hardware limit to the
    model's score, and it does so *selectively*: batches are length-sorted, so
    the examples that OOM are the LONGEST ones. Dropping them silently biases
    every metric toward short, easy postings.

    Measured on this box before the fix: an arm scoring 94% compliance on its
    first batches reported 55% overall, because roughly 40% of examples -- all
    of them long -- generated nothing at all. That number would have gone into
    the README as a model result.

    So: halve and retry. Only a single example that still cannot fit is recorded
    as empty, and that is a genuine, reportable limit rather than an artifact.
    """
    retried_empty = False
    while True:
        size = max(1, min(cap[0], len(chunk)))
        try:
            texts: list[str] = []
            for s in range(0, len(chunk), size):
                sub = chunk[s : s + size]
                if size > 1:
                    texts.extend(backend.generate_batch([e["source_text"] for e in sub], few_shot))
                else:
                    texts.append(backend.generate(sub[0]["source_text"], few_shot))

            # An entire batch coming back empty is not a model result, it is the
            # signature of a GPU that has stopped generating. Observed on this
            # 4 GB card: ~280 examples into a length-sorted run, every remaining
            # batch returned zero new tokens instantly and no exception was
            # raised anywhere. The same examples generate correctly in a fresh
            # process, so the cause is accumulated allocator state rather than
            # anything about the inputs.
            #
            # Handled like an OOM because the remedy is the same: free the cache,
            # come back smaller. Retried at most once per chunk, so a model that
            # genuinely produces nothing for one hard batch is still scored as
            # having produced nothing.
            if texts and not any(t.strip() for t in texts) and not retried_empty:
                retried_empty = True
                _free_cuda()
                if size > 1:
                    cap[0] = max(1, size // 2)
                LOGGER.warning("whole batch of %d empty near %s; cleared CUDA cache, "
                               "retrying at %d", len(texts), chunk[0]["posting_id"], cap[0])
                continue
            return texts
        except Exception as e:
            if not _is_oom(e):
                LOGGER.warning("generation failed near %s: %s", chunk[0]["posting_id"], e)
                return [""] * len(chunk)
            _free_cuda()
            if size <= 1:
                # One example alone does not fit. Genuinely unservable on this
                # card; report it rather than pretending otherwise.
                LOGGER.warning("OOM at batch size 1 near %s; recording empty",
                               chunk[0]["posting_id"])
                return [""] * len(chunk)
            cap[0] = size // 2
            LOGGER.warning("OOM at batch size %d; retrying at %d for the rest of the run",
                           size, cap[0])


#: An arm that produces this fraction of *instant, empty* completions is
#: reporting the machine's state, not the model's accuracy.
EMPTY_ARM_THRESHOLD = 0.10
INSTANT_SECONDS = 0.05


def assert_generation_sane(name: str, completions: list[tuple[str, float]]) -> None:
    """Refuse to score a run that clearly did not generate. Raises SystemExit.

    Why a hard stop rather than a warning
    -------------------------------------
    This harness has now twice produced a plausible-looking headline number out
    of a broken machine state: 55% compliance when batches were silently
    OOM-dropped, and 48.6% when generation returned instantly with empty strings
    and raised nothing at all. Both times the model was fine -- re-running the
    same examples afterwards produced valid JSON for every one of them. Both
    numbers were within the range a mediocre model might genuinely score, so
    nothing about the value itself looked wrong, and either could have been
    written into the README as a result.

    What distinguishes a broken run from a bad model is not the score, it is the
    SHAPE: a model that fails a hard example still spends time on it and still
    emits tokens. Empty output returned in ~0 seconds means no forward pass
    happened. That is the signal checked here.

    A genuinely weak model emitting prose instead of JSON is untouched by this:
    its completions are non-empty and slow, and it is scored as non-compliant,
    which is the correct outcome.
    """
    if not completions:
        return
    instant_empty = sum(1 for text, dt in completions
                        if not text.strip() and dt < INSTANT_SECONDS)
    share = instant_empty / len(completions)
    if share <= EMPTY_ARM_THRESHOLD:
        if instant_empty:
            LOGGER.warning("%d/%d completions were empty and instant (%.1f%%) -- below the "
                           "%.0f%% abort threshold, but worth a look",
                           instant_empty, len(completions), 100 * share,
                           100 * EMPTY_ARM_THRESHOLD)
        return
    raise SystemExit(
        f"\nABORTING {name}: {instant_empty}/{len(completions)} completions "
        f"({share:.1%}) were empty AND returned in under {INSTANT_SECONDS}s.\n"
        "No forward pass happened for those examples, so any score computed here "
        "would describe this machine, not the model.\n\n"
        "Usual cause: another process still holds VRAM, leaving the GPU in a state "
        "where generate() returns nothing without raising.\n"
        "  1. close other GPU users, then confirm the memory is actually free:\n"
        "       nvidia-smi\n"
        "  2. re-run this arm\n"
        "If it recurs on a genuinely idle GPU, lower --batch-size."
    )


def run(backend: Backend, examples: list[dict[str, Any]], few_shot: list[tuple[str, str]],
        limit: int | None = None) -> dict[str, Any]:
    """Generate + score. Returns a payload safe to cache to disk and re-score."""
    if limit:
        examples = examples[:limit]

    results = []
    preds: dict[str, Any] = {}
    golds: dict[str, Any] = {}
    raw_rows: list[dict[str, Any]] = []

    completions = generate_completions(backend, examples, few_shot)
    assert_generation_sane(backend.name, completions)
    for i, (ex, (out, dt)) in enumerate(zip(examples, completions, strict=True), 1):
        res = score_example(ex["posting_id"], ex["source_text"], ex["target_json"], out, latency_s=dt)
        results.append(res)
        pred, _ = parse_prediction(out)
        preds[ex["posting_id"]] = pred
        gold, _ = parse_prediction(ex["target_json"])
        golds[ex["posting_id"]] = gold
        raw_rows.append({
            "posting_id": ex["posting_id"],
            "raw_output": out,
            "gold_json": ex["target_json"],
            # A source excerpt travels with the prediction so failure analysis is
            # readable without re-joining against the dataset -- and so a saved
            # prediction file is self-contained evidence.
            "source_excerpt": ex["source_text"][:1200],
            "latency_s": dt,
            "parse_error": res.parse_error,
            "exact_match": res.exact_match,
            "ungrounded": res.ungrounded,
        })
        if i % 50 == 0:
            LOGGER.info("scored %d/%d | compliance so far %.1f%%", i, len(examples),
                        100 * sum(r.lenient_compliant for r in results) / len(results))

    metrics = aggregate(results, golds, preds)
    ci = bootstrap_ci(results, golds, preds)
    return {
        "backend": backend.name,
        "n": len(results),
        "metrics": metrics,
        "ci95": {k: list(v) for k, v in ci.items()},
        "per_example": raw_rows,
        "results": [asdict(r) for r in results],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["hf", "gguf", "gemini"], required=True)
    ap.add_argument("--adapter", default=None, help="path to a PEFT adapter directory")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--n-gpu-layers", type=int, default=0)
    ap.add_argument("--gemini-model", default="gemini-2.0-flash")
    ap.add_argument("--unconstrained-gemini", action="store_true",
                    help="disable Gemini's response_schema for the apples-to-apples row")
    ap.add_argument("--fp16", action="store_true", help="load fp16 instead of 4-bit (needs >6 GB VRAM)")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="hf backend only. 8 fits a 4 GB card at seq 2048; drop to 4 if you OOM")
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--few-shot", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N (debugging)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from config import setup_run

    setup_run()

    if args.split == "test":
        LOGGER.info("evaluating on the HELD-OUT TEST SPLIT -- results are final, do not tune against them")

    examples = load_split(args.data_dir, args.split)
    few_shot = load_fewshot(args.data_dir, args.few_shot)
    LOGGER.info("split=%s n=%d few_shot=%d", args.split, len(examples), len(few_shot))

    backend = build_backend(args)
    payload = run(backend, examples, few_shot, limit=args.limit)
    payload["config"] = {
        "split": args.split, "few_shot": args.few_shot, "adapter": args.adapter,
        "gguf": args.gguf, "max_new_tokens": MAX_NEW_TOKENS, "max_source_chars": MAX_SOURCE_CHARS,
    }
    # Pull rank/alpha straight out of the adapter so the ablation table reports
    # what was actually loaded, not what the filename claims.
    if args.adapter:
        acfg_path = Path(args.adapter) / "adapter_config.json"
        if acfg_path.exists():
            acfg = json.loads(acfg_path.read_text(encoding="utf-8"))
            payload["config"].update({
                "lora_rank": acfg.get("r"),
                "lora_alpha": acfg.get("lora_alpha"),
                "target_modules": acfg.get("target_modules"),
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    m = payload["metrics"]
    LOGGER.info(
        "%s | compliance(strict/lenient) %.1f%%/%.1f%% | exact %.1f%% | F1_micro %.3f | "
        "hallucination %.1f%% | %.2fs/ex",
        backend.name, 100 * m["schema_compliance_strict"], 100 * m["schema_compliance_lenient"],
        100 * m["exact_match"], m["field_f1_micro"], 100 * m["hallucination_rate"], m["mean_latency_s"],
    )
    LOGGER.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

