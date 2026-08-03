"""Build the head-to-head comparison table from saved prediction files.

    python -m eval.compare_models \
        --run "Base (0-shot)=results/raw_predictions/base_0shot.json" \
        --run "Base (3-shot)=results/raw_predictions/base_3shot.json" \
        --run "LoRA r=16 (ours)=results/raw_predictions/finetuned_r16.json" \
        --run "Gemini 2.0 Flash=results/raw_predictions/gemini.json" \
        --baseline "Base (0-shot)" --ceiling "Gemini 2.0 Flash" --ours "LoRA r=16 (ours)" \
        --out results/comparison_table.md

This script deliberately consumes *saved predictions* rather than re-running
models. Generation is the expensive, non-deterministic-ish part; scoring is cheap
and pure. Separating them means the table can be rebuilt after a metric bug fix
without spending another hour of GPU time or another 500 Gemini calls -- and it
means the raw completions stay on disk as evidence.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from data.schema import FIELD_SPECS, TIER_NAMES, parse_prediction
from eval.metrics import HEADLINE_METRICS, ExampleResult, paired_bootstrap_pvalue

LOGGER = logging.getLogger("compare")

#: Metrics where lower is better -- the table marks these so a reader does not
#: have to remember which direction each column runs.
LOWER_IS_BETTER = {"hallucination_rate", "over_emission_rate"}

PRETTY = {
    "schema_compliance_strict": "Schema compliance (strict)",
    "schema_compliance_lenient": "Schema compliance (lenient)",
    "exact_match": "Exact match",
    "field_f1_micro": "Field F1 (micro)",
    "field_f1_macro": "Field F1 (macro)",
    "hallucination_rate": "Hallucination rate ↓",
    "over_emission_rate": "Over-emission rate ↓",
    "null_recall": "Null recall",
}


def load_run(spec: str) -> tuple[str, dict[str, Any]]:
    """Parse a `Label=path.json` CLI argument."""
    if "=" not in spec:
        raise SystemExit(f"--run needs LABEL=PATH, got {spec!r}")
    label, path = spec.split("=", 1)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return label.strip(), payload


def _rehydrate(payload: dict[str, Any]) -> tuple[list[ExampleResult], dict[str, Any], dict[str, Any]]:
    """Rebuild the objects the significance test needs from a saved run."""
    results = [ExampleResult(**r) for r in payload["results"]]
    preds: dict[str, Any] = {}
    golds: dict[str, Any] = {}
    for row in payload["per_example"]:
        preds[row["posting_id"]], _ = parse_prediction(row["raw_output"])
        golds[row["posting_id"]], _ = parse_prediction(row["gold_json"])
    return results, golds, preds


def gap_closed(baseline: float, ours: float, ceiling: float, lower_better: bool = False) -> float | None:
    """Fraction of the baseline->ceiling gap that the fine-tuned model closes.

    Returns None when the gap is degenerate (ceiling not better than baseline),
    because "closed 400% of the gap" is a meaningless statistic that shows up
    whenever the denominator is near zero -- and it is exactly the kind of number
    that gets a project dismissed. If our model *exceeds* the ceiling the value
    is >1.0, which is reported honestly rather than clipped.
    """
    gap = (baseline - ceiling) if lower_better else (ceiling - baseline)
    if gap <= 1e-9:
        return None
    got = (baseline - ours) if lower_better else (ours - baseline)
    return got / gap


def build_table(runs: list[tuple[str, dict[str, Any]]], baseline: str | None,
                ceiling: str | None, ours: str | None) -> str:
    labels = [lab for lab, _ in runs]
    by_label = dict(runs)
    n = {lab: p["n"] for lab, p in runs}

    lines = [
        "# Model comparison",
        "",
        f"Held-out test split, n = {n[labels[0]]} postings, identical prompt template for every arm "
        "(`data/schema.py`), greedy decoding, `max_new_tokens=400`.",
        "",
        "Brackets are 95% bootstrap confidence intervals (1000 resamples at the **example** level).",
        "",
        "| Metric | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(labels),
    ]

    for metric in HEADLINE_METRICS:
        cells = []
        for lab in labels:
            m = by_label[lab]["metrics"].get(metric)
            ci = by_label[lab].get("ci95", {}).get(metric)
            if m is None:
                cells.append("-")
                continue
            cell = f"{m:.1%}" if metric != "field_f1_micro" and metric != "field_f1_macro" else f"{m:.3f}"
            if ci:
                lo, hi = ci
                fmt = (lambda x: f"{x:.1%}") if "f1" not in metric else (lambda x: f"{x:.3f}")
                cell += f"<br><sub>[{fmt(lo)}, {fmt(hi)}]</sub>"
            cells.append(cell)
        lines.append(f"| {PRETTY.get(metric, metric)} | " + " | ".join(cells) + " |")

    lines.append("| Mean latency / example | " + " | ".join(
        f"{by_label[lab]['metrics'].get('mean_latency_s', float('nan')):.2f}s" for lab in labels) + " |")

    # ---- the headline sentence -------------------------------------------
    if baseline and ceiling and ours and all(k in by_label for k in (baseline, ceiling, ours)):
        lines += ["", "## Gap closed to the ceiling", "",
                  f"How much of the `{baseline}` → `{ceiling}` gap the fine-tuned model recovers.", "",
                  "| Metric | " + f"{baseline} | {ours} | {ceiling} | gap closed |", "|---|---:|---:|---:|---:|"]
        for metric in ("schema_compliance_lenient", "field_f1_micro", "exact_match", "hallucination_rate"):
            b = by_label[baseline]["metrics"].get(metric)
            o = by_label[ours]["metrics"].get(metric)
            c = by_label[ceiling]["metrics"].get(metric)
            if None in (b, o, c):
                continue
            g = gap_closed(b, o, c, lower_better=metric in LOWER_IS_BETTER)
            gtxt = "n/a (no gap)" if g is None else f"**{g:.0%}**"
            f = (lambda x: f"{x:.3f}") if "f1" in metric else (lambda x: f"{x:.1%}")
            lines.append(f"| {PRETTY.get(metric, metric)} | {f(b)} | {f(o)} | {f(c)} | {gtxt} |")

        lines += ["", "> **Caveat, stated up front:** the gold labels were produced by the same "
                  f"model family as the `{ceiling}` arm, so that column is biased upward and the "
                  "gap-closed percentages are therefore *conservative* — the true gap is smaller "
                  "than the one shown. `results/label_audit.md` quantifies label quality against "
                  "LinkedIn's own non-LLM metadata, which is independent of any model."]

    # ---- significance ----------------------------------------------------
    if ours and baseline and ours in by_label and baseline in by_label:
        LOGGER.info("running paired bootstrap significance test (this takes a minute)")
        ra, ga, pa = _rehydrate(by_label[ours])
        rb, _gb, pb = _rehydrate(by_label[baseline])
        pval = paired_bootstrap_pvalue(ra, rb, ga, pa, pb, metric="field_f1_micro")
        lines += ["", "## Significance", "",
                  f"Paired bootstrap on `field_f1_micro`, `{ours}` vs `{baseline}`: "
                  f"**p = {pval:.4f}** (1000 resamples, two-sided).", "",
                  "Paired because both models are scored on the same postings; an unpaired test "
                  "would ignore that shared difficulty and understate the effect. Note that "
                  "overlapping confidence intervals in the table above do **not** imply "
                  "non-significance — this test is the one to read."]

    # ---- per-field breakdown ---------------------------------------------
    lines += ["", "## Per-field F1", "",
              "Grouped by difficulty tier. `support` is the number of non-null gold values in the "
              "test split — a field with low support has a wide interval and should not carry an "
              "argument on its own.", "",
              "| Field | Tier | support | " + " | ".join(labels) + " |",
              "|---|---|---:|" + "---:|" * len(labels)]
    for fname, spec in sorted(FIELD_SPECS.items(), key=lambda kv: (kv[1]["tier"], kv[0])):
        first = by_label[labels[0]]["metrics"]["per_field"].get(fname, {})
        cells = [f"{by_label[lab]['metrics']['per_field'].get(fname, {}).get('f1', 0):.3f}" for lab in labels]
        lines.append(f"| `{fname}` | {spec['tier']} ({TIER_NAMES[spec['tier']]}) | "
                     f"{first.get('support', 0)} | " + " | ".join(cells) + " |")

    lines += ["", "## Per-tier mean F1", "", "| Tier | " + " | ".join(labels) + " |",
              "|---|" + "---:|" * len(labels)]
    tiers = by_label[labels[0]]["metrics"].get("per_tier", {})
    for tkey in tiers:
        cells = [f"{by_label[lab]['metrics'].get('per_tier', {}).get(tkey, 0):.3f}" for lab in labels]
        lines.append(f"| {tkey.replace('_', ' ')} | " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--ceiling", default=None)
    ap.add_argument("--ours", default=None)
    ap.add_argument("--out", type=Path, default=Path("results/comparison_table.md"))
    ap.add_argument("--json-out", type=Path, default=Path("results/comparison.json"))
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    runs = [load_run(s) for s in args.run]
    ns = {lab: p["n"] for lab, p in runs}
    if len(set(ns.values())) != 1:
        # Comparing arms scored on different numbers of examples is the single
        # easiest way to publish a misleading table. Refuse rather than warn.
        raise SystemExit(f"arms were evaluated on different example counts: {ns}. Re-run them on the same split.")

    table = build_table(runs, args.baseline, args.ceiling, args.ours)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table, encoding="utf-8")
    LOGGER.info("wrote %s", args.out)

    # A machine-readable copy so generate_report.py does not re-parse markdown.
    args.json_out.write_text(json.dumps(
        {lab: {"metrics": p["metrics"], "ci95": p.get("ci95", {}), "n": p["n"]} for lab, p in runs},
        indent=2), encoding="utf-8")
    LOGGER.info("wrote %s", args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
