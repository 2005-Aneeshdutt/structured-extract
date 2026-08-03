"""Does Q4_K_M quantization degrade the fine-tuned model? Measure, do not assume.

    python -m quantize.verify_quantized \
        --gguf models/qwen2.5-1.5b-r16-a32-Q4_K_M.gguf \
        --adapter outputs/qwen2.5-1.5b-r16-a32/adapter \
        --n 50 --split val

Why this script exists
----------------------
"I quantized it to Q4_K_M" is a claim about a file. "The quantized model performs
the same" is a claim about behavior, and nothing about the first implies the
second. Structured JSON output is unusually sensitive to quantization: the model
must emit exactly balanced braces and exact enum spellings, and a small shift in
logits at a brace token turns a perfect extraction into an unparseable string.
So the delta is measured on real examples and reported with its sign.

**Run this on the VALIDATION split, not test.** If a regression shows up you will
want to try Q5_K_M or Q8_0 and re-measure, and iterating against the test split
is exactly how a held-out set stops being held out. The test split is touched
once, at the end, for the final table.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from data.schema import FIELD_SPECS, flatten, parse_prediction
from eval.metrics import aggregate, score_example, values_equal
from eval.run_eval import GGUFBackend, HFBackend, load_split

LOGGER = logging.getLogger("verify")

#: A metric drop larger than this is reported as a regression rather than noise.
#: 2pp on n=50 is comfortably inside binomial noise (SE ~= 7pp on a rate near
#: 0.5), so this threshold is intentionally a *screening* tool: it flags things
#: worth a bigger run, it does not certify equivalence. The script says so in its
#: own output rather than letting a reader over-read n=50.
REGRESSION_PP = 0.02


def compare(fp_payload: dict[str, Any], q_payload: dict[str, Any]) -> dict[str, Any]:
    """Per-metric delta plus a per-example agreement rate.

    Agreement (do the two models produce the *same* object, right or wrong) is
    the more sensitive instrument here. Aggregate metrics can match while the two
    models disagree on 30% of examples in offsetting directions; agreement
    catches that, and it is the number that tells you whether the GGUF is really
    the same model.
    """
    out: dict[str, Any] = {"metrics": {}}
    for key in ("schema_compliance_lenient", "exact_match", "field_f1_micro",
                "hallucination_rate", "mean_latency_s"):
        a = fp_payload["metrics"].get(key, 0.0)
        b = q_payload["metrics"].get(key, 0.0)
        out["metrics"][key] = {"fp16_adapter": a, "gguf_q4": b, "delta": b - a}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--split", default="val", choices=["val", "train"],
                    help="test is intentionally not offered here -- see the module docstring")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--n-gpu-layers", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/quantization_check.md"))
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    examples = load_split(args.data_dir, args.split)[: args.n]
    LOGGER.info("verifying on %d %s examples", len(examples), args.split)

    payloads: dict[str, dict[str, Any]] = {}
    outputs: dict[str, dict[str, str]] = {}

    for tag, backend in (
        ("fp16_adapter", HFBackend(adapter=args.adapter, model_id=args.base_model)),
        ("gguf_q4", GGUFBackend(args.gguf, n_gpu_layers=args.n_gpu_layers)),
    ):
        LOGGER.info("running %s (%s)", tag, backend.name)
        results, preds, golds, texts = [], {}, {}, {}
        for ex in examples:
            import time

            t0 = time.perf_counter()
            raw = backend.generate(ex["source_text"], None)
            dt = time.perf_counter() - t0
            results.append(score_example(ex["posting_id"], ex["source_text"],
                                         ex["target_json"], raw, latency_s=dt))
            preds[ex["posting_id"]], _ = parse_prediction(raw)
            golds[ex["posting_id"]], _ = parse_prediction(ex["target_json"])
            texts[ex["posting_id"]] = raw
        payloads[tag] = {"metrics": aggregate(results, golds, preds)}
        outputs[tag] = texts
        # Free the weights before loading the next backend: 4 GB of VRAM will not
        # hold a 4-bit transformer and a llama.cpp context at the same time if
        # n_gpu_layers > 0.
        del backend
        try:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except ImportError:
            pass

    # ---- per-example agreement -------------------------------------------
    n_same_obj = 0
    field_disagreements: dict[str, int] = dict.fromkeys(FIELD_SPECS, 0)
    for ex in examples:
        pid = ex["posting_id"]
        a, _ = parse_prediction(outputs["fp16_adapter"][pid])
        b, _ = parse_prediction(outputs["gguf_q4"][pid])
        if a is None or b is None:
            if a is None and b is None:
                n_same_obj += 1
            continue
        fa, fb = flatten(a), flatten(b)
        same = True
        for k in FIELD_SPECS:
            if not values_equal(k, fa[k], fb[k]):
                field_disagreements[k] += 1
                same = False
        n_same_obj += int(same)

    cmp = compare(payloads["fp16_adapter"], payloads["gguf_q4"])
    agreement = n_same_obj / max(len(examples), 1)

    regressions = [k for k, v in cmp["metrics"].items()
                   if k != "mean_latency_s"
                   and (v["delta"] < -REGRESSION_PP if k != "hallucination_rate"
                        else v["delta"] > REGRESSION_PP)]

    lines = [
        "# Quantization verification: fp16 + LoRA adapter vs GGUF Q4_K_M",
        "",
        f"Split: `{args.split}` · n = {len(examples)} · GGUF: `{Path(args.gguf).name}` "
        f"(`n_gpu_layers={args.n_gpu_layers}`)",
        "",
        "| metric | fp16 + adapter | GGUF Q4_K_M | delta |",
        "|---|---:|---:|---:|",
    ]
    for k, v in cmp["metrics"].items():
        fmt = (lambda x: f"{x:.2f}s") if k == "mean_latency_s" else (
            (lambda x: f"{x:.3f}") if "f1" in k else (lambda x: f"{x:.1%}"))
        sign = "+" if v["delta"] >= 0 else ""
        dfmt = f"{sign}{v['delta']:.2f}s" if k == "mean_latency_s" else (
            f"{sign}{v['delta']:.3f}" if "f1" in k else f"{sign}{v['delta']:.1%}")
        lines.append(f"| `{k}` | {fmt(v['fp16_adapter'])} | {fmt(v['gguf_q4'])} | {dfmt} |")

    lines += [
        "",
        f"**Object-level agreement:** the two backends produce an identical parsed object on "
        f"**{agreement:.0%}** of examples.",
        "",
        "Agreement is the more sensitive check: aggregate scores can match while the two "
        "models disagree on many examples in offsetting directions.",
        "",
        "## Fields where the two backends disagree most",
        "",
        "| field | disagreements |",
        "|---|---:|",
    ]
    for k, c in sorted(field_disagreements.items(), key=lambda kv: -kv[1])[:8]:
        if c:
            lines.append(f"| `{k}` | {c} |")

    verdict = ("**No regression detected** at the "
               f"{REGRESSION_PP:.0%} screening threshold." if not regressions
               else f"**Regression flagged** on: {', '.join(f'`{r}`' for r in regressions)}.")
    lines += [
        "",
        "## Verdict",
        "",
        verdict,
        "",
        f"Read this as a screen, not a certification: at n={len(examples)} the binomial standard "
        "error on a rate near 0.5 is about 7pp, so this run can only catch a *large* regression. "
        "It is sized to be cheap enough to run after every quantization change. If it flags "
        "something, re-run at n=500 before concluding anything; if it passes, the honest "
        "statement is 'no large regression', not 'identical'.",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("wrote %s", args.out)
    LOGGER.info("agreement=%.0f%% regressions=%s", 100 * agreement, regressions or "none")
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
