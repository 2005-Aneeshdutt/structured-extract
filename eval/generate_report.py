"""Turn saved prediction files into the README's charts, ablation table and
failure analysis.

    python -m eval.generate_report \
        --run "Base 0-shot=results/raw_predictions/base_0shot.json" \
        --run "Base 3-shot=results/raw_predictions/base_3shot.json" \
        --run "LoRA r=16 (ours)=results/raw_predictions/finetuned_r16.json" \
        --run "Gemini 2.0 Flash=results/raw_predictions/gemini.json" \
        --ours "LoRA r=16 (ours)" --baseline "Base 0-shot" \
        --ablation "r=8=results/raw_predictions/val_r8.json" \
        --ablation "r=16=results/raw_predictions/val_r16.json" \
        --ablation "r=32=results/raw_predictions/val_r32.json" \
        --robustness results/raw_predictions/robustness_finetuned.json \
        --robustness-baseline results/raw_predictions/robustness_base.json

Outputs into results/: charts/*.png (light + dark), ablation_table.md,
failure_analysis.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from data.schema import FIELD_SPECS, flatten, parse_prediction
from eval.charts import grouped_bars, horizontal_bars, save_both_themes
from eval.compare_models import load_run

LOGGER = logging.getLogger("report")

HEADLINE_FOR_CHART = [
    ("schema_compliance_lenient", "Schema compliance"),
    ("field_f1_micro", "Field F1 (micro)"),
    ("exact_match", "Exact match"),
    ("null_recall", "Null recall"),
]


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def chart_model_comparison(runs: list[tuple[str, dict[str, Any]]], out_dir: Path) -> None:
    cats = [label for _k, label in HEADLINE_FOR_CHART]
    series = {lab: [p["metrics"].get(k, 0.0) for k, _ in HEADLINE_FOR_CHART] for lab, p in runs}
    errors = {
        lab: [tuple(p.get("ci95", {}).get(k, (p["metrics"].get(k, 0.0),) * 2)) for k, _ in HEADLINE_FOR_CHART]
        for lab, p in runs
    }
    n = runs[0][1]["n"]
    save_both_themes(
        lambda th: grouped_bars(
            cats, series, th,
            title="Structured JSON extraction: base vs LoRA fine-tune vs ceiling",
            subtitle=f"Held-out test split, n={n} · identical prompt for every arm · whiskers are 95% bootstrap CI",
            ylabel="score", errors=errors),
        out_dir, "model_comparison")


def chart_hallucination(runs: list[tuple[str, dict[str, Any]]], out_dir: Path) -> None:
    """Hallucination and over-emission get their own chart.

    Separated from the headline chart because lower is better here. Putting
    inverted-polarity metrics in the same panel as higher-is-better ones is a
    reliable way to make a reader draw the opposite conclusion at a glance.
    """
    cats = ["Hallucination rate", "Over-emission rate"]
    series = {lab: [p["metrics"].get("hallucination_rate", 0.0),
                    p["metrics"].get("over_emission_rate", 0.0)] for lab, p in runs}
    top = max(max(v) for v in series.values()) if series else 1.0
    save_both_themes(
        lambda th: grouped_bars(
            cats, series, th,
            title="Unsupported output — lower is better",
            subtitle="Share of emitted values with no support in the source text (left) "
                     "and where the gold value is null (right)",
            ylabel="rate", ylim=(0.0, min(1.0, max(0.2, top * 1.25)))),
        out_dir, "hallucination")


def chart_per_field(runs: list[tuple[str, dict[str, Any]]], out_dir: Path,
                    keep: list[str] | None = None) -> None:
    labels = keep or [lab for lab, _ in runs]
    by = dict(runs)
    fields = sorted(FIELD_SPECS, key=lambda f: (FIELD_SPECS[f]["tier"], f))
    ticks = [f"{f}  ·  T{FIELD_SPECS[f]['tier']}" for f in fields]
    series = {lab: [by[lab]["metrics"]["per_field"].get(f, {}).get("f1", 0.0) for f in fields]
              for lab in labels}
    save_both_themes(
        lambda th: horizontal_bars(
            ticks, series, th,
            title="Per-field F1 on the held-out test split",
            subtitle="Ordered by difficulty tier: T1 verbatim · T2 nested · T3 closed-vocabulary/set · T4 normalization + restraint"),
        out_dir, "per_field_f1")


def chart_ablation(ablation: list[tuple[str, dict[str, Any]]], out_dir: Path) -> None:
    cats = ["Schema compliance", "Field F1 (micro)", "Exact match"]
    keys = ["schema_compliance_lenient", "field_f1_micro", "exact_match"]
    series = {lab: [p["metrics"].get(k, 0.0) for k in keys] for lab, p in ablation}
    errors = {lab: [tuple(p.get("ci95", {}).get(k, (p["metrics"].get(k, 0.0),) * 2)) for k in keys]
              for lab, p in ablation}
    save_both_themes(
        lambda th: grouped_bars(
            cats, series, th,
            title="LoRA rank ablation",
            subtitle="Validation split — the test split is not used for model selection",
            ylabel="score", errors=errors),
        out_dir, "ablation")


def chart_robustness(ours: dict[str, Any], baseline: dict[str, Any] | None, out_dir: Path) -> None:
    arms = list(ours["arms"])
    series: dict[str, list[float]] = {}
    if baseline:
        series["Base 0-shot"] = [baseline["arms"].get(a, {}).get("metrics", {}).get("field_f1_micro", 0.0)
                                 for a in arms]
    series["LoRA fine-tuned"] = [ours["arms"][a]["metrics"].get("field_f1_micro", 0.0) for a in arms]
    save_both_themes(
        lambda th: horizontal_bars(
            arms, series, th,
            title="Robustness to perturbed inputs",
            subtitle="Field F1 (micro) per perturbation · every perturbation also transforms the gold label, "
                     "so a drop is brittleness and not label noise",
            figsize=(10.0, 6.0)),
        out_dir, "robustness")


# ---------------------------------------------------------------------------
# Ablation table
# ---------------------------------------------------------------------------


def ablation_table(ablation: list[tuple[str, dict[str, Any]]]) -> str:
    lines = [
        "# LoRA rank ablation",
        "",
        "Measured on the **validation** split. The test split is never used for model "
        "selection — that is what keeps the headline table honest.",
        "",
        "All three runs are identical except for `lora_rank`, with `alpha = 2 x rank` so the "
        "effective LoRA scaling stays fixed at 2.0. Without that, changing rank would change "
        "capacity *and* effective learning rate at once, and the comparison would not isolate "
        "anything.",
        "",
        "| rank | alpha | trainable params | schema compliance | field F1 (micro) | exact match | hallucination ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for lab, p in ablation:
        m = p["metrics"]
        cfg = p.get("config", {})
        lines.append(
            f"| {lab} | {cfg.get('lora_alpha', '—')} | {cfg.get('trainable_params', '—')} | "
            f"{m.get('schema_compliance_lenient', 0):.1%} | {m.get('field_f1_micro', 0):.3f} | "
            f"{m.get('exact_match', 0):.1%} | {m.get('hallucination_rate', 0):.1%} |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "The interesting question is not which number is biggest — it is whether the "
        "differences exceed the bootstrap interval. If rank 8 is within the CI of rank 32, "
        "the correct conclusion is that **this task does not need the extra capacity**, and "
        "the smaller adapter ships. Reporting a 0.3pp 'win' for rank 32 as if it were real "
        "is the mistake this table exists to prevent.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Failure analysis
# ---------------------------------------------------------------------------

FAILURE_CATEGORIES = {
    "unparseable": "Output was not valid JSON / did not validate against the schema",
    "salary_misparse": "Salary present but a bound, currency or period is wrong",
    "over_emission": "Asserted a value where the source says nothing (gold null)",
    "missed_extraction": "Left a field null that the source clearly states",
    "enum_confusion": "Closed-vocabulary field mapped to the wrong member",
    "set_partial": "Skills / benefits list partially wrong (missing or extra items)",
    "ungrounded": "Emitted a value with no supporting span in the source",
}


def categorize(row: dict[str, Any]) -> list[str]:
    """Assign failure categories to one example. An example can carry several."""
    cats: list[str] = []
    if row.get("parse_error"):
        return ["unparseable"]
    pred, _ = parse_prediction(row["raw_output"])
    gold, _ = parse_prediction(row["gold_json"])
    if pred is None or gold is None:
        return ["unparseable"]
    pf, gf = flatten(pred), flatten(gold)
    if row.get("ungrounded"):
        cats.append("ungrounded")
    for k, spec in FIELD_SPECS.items():
        g, p = gf[k], pf[k]
        if g == p:
            continue
        if k.startswith("salary."):
            cats.append("salary_misparse")
        elif g in (None, []) and p not in (None, []):
            cats.append("over_emission")
        elif g not in (None, []) and p in (None, []):
            cats.append("missed_extraction")
        elif spec["comparator"] == "categorical":
            cats.append("enum_confusion")
        elif spec["comparator"] == "set":
            cats.append("set_partial")
    return sorted(set(cats))


def failure_analysis(payload: dict[str, Any], n_examples: int = 10, excerpt: int = 700) -> str:
    """Pick failing examples spread across categories, not the first N.

    Taking the first 10 failures would over-represent whatever category happens to
    be common, which tells a reader nothing they could not get from the aggregate
    table. Sampling one per category first, then filling, surfaces the long tail —
    which is where the actionable next steps live.
    """
    rows = payload["per_example"]
    failing = [r for r in rows if not r.get("exact_match")]
    tagged = [(r, categorize(r)) for r in failing]
    counts: Counter[str] = Counter(c for _r, cs in tagged for c in cs)

    chosen: list[tuple[dict[str, Any], list[str]]] = []
    seen_ids: set[str] = set()
    for cat in FAILURE_CATEGORIES:
        for r, cs in tagged:
            if cat in cs and r["posting_id"] not in seen_ids:
                chosen.append((r, cs))
                seen_ids.add(r["posting_id"])
                break
    for r, cs in tagged:
        if len(chosen) >= n_examples:
            break
        if r["posting_id"] not in seen_ids:
            chosen.append((r, cs))
            seen_ids.add(r["posting_id"])

    lines = [
        "# Failure analysis",
        "",
        f"Model: `{payload.get('backend', 'unknown')}` · {len(failing)} of {len(rows)} test examples "
        f"({len(failing) / max(len(rows), 1):.1%}) are not an exact match on all 18 leaf fields.",
        "",
        "Exact match is a deliberately brutal metric — one wrong skill in a 12-item list fails the "
        "whole example. The per-field table in `comparison_table.md` is the fairer view; this "
        "document exists to characterise *what kind* of wrong the remaining errors are.",
        "",
        "## Failure taxonomy",
        "",
        "| category | description | examples affected | share of failures |",
        "|---|---|---:|---:|",
    ]
    for cat, desc in FAILURE_CATEGORIES.items():
        c = counts.get(cat, 0)
        lines.append(f"| `{cat}` | {desc} | {c} | {c / max(len(failing), 1):.1%} |")

    lines += ["", "## Worked examples", ""]
    for i, (r, cs) in enumerate(chosen[:n_examples], 1):
        pred, _ = parse_prediction(r["raw_output"])
        gold, _ = parse_prediction(r["gold_json"])
        lines += [
            f"### {i}. `{r['posting_id']}` — {', '.join(f'`{c}`' for c in cs) or 'uncategorized'}",
            "",
            "<details><summary>source excerpt</summary>",
            "",
            "```text",
            (r.get("source_excerpt") or "(source text not stored in this prediction file)")[:excerpt],
            "```",
            "",
            "</details>",
            "",
            "| field | gold | predicted |",
            "|---|---|---|",
        ]
        if pred is None or gold is None:
            lines += [f"| — | — | model output did not parse: `{r.get('parse_error')}` |", ""]
            lines += ["```text", r["raw_output"][:400], "```", ""]
            continue
        pf, gf = flatten(pred), flatten(gold)
        diffs = [k for k in FIELD_SPECS if gf[k] != pf[k]]
        for k in diffs[:12]:
            lines.append(f"| `{k}` | `{json.dumps(gf[k], ensure_ascii=False)}` | "
                         f"`{json.dumps(pf[k], ensure_ascii=False)}` |")
        if r.get("ungrounded"):
            lines.append(f"| _ungrounded_ | — | `{', '.join(r['ungrounded'])}` |")
        lines.append("")

    lines += [
        "## What this suggests",
        "",
        "Fill this in once real numbers exist — but the analysis to write is: which category "
        "dominates, and is the fix more data, a schema change, or constrained decoding? "
        "If `unparseable` dominates, grammar-constrained decoding (llama.cpp GBNF) removes it "
        "outright and is a stronger answer than more training. If `over_emission` dominates, "
        "the training mix needs more negative examples. If `salary_misparse` dominates, the "
        "normalization convention itself may be underspecified in the prompt.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], metavar="LABEL=PATH")
    ap.add_argument("--ablation", action="append", default=[], metavar="LABEL=PATH")
    ap.add_argument("--robustness", type=Path, default=None)
    ap.add_argument("--robustness-baseline", type=Path, default=None)
    ap.add_argument("--ours", default=None)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--n-failures", type=int, default=10)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    charts_dir = args.results_dir / "charts"

    if args.run:
        runs = [load_run(s) for s in args.run]
        chart_model_comparison(runs, charts_dir)
        chart_hallucination(runs, charts_dir)
        keep = [lab for lab in (args.baseline, args.ours) if lab] or None
        if keep and len(keep) >= 2:
            # Per-field chart shows base vs ours only: 18 categories x 4 series is
            # 72 bars, which is unreadable at README width.
            chart_per_field(runs, charts_dir, keep=keep)
        else:
            chart_per_field(runs, charts_dir)
        LOGGER.info("wrote comparison charts -> %s", charts_dir)

        if args.ours:
            by = dict(runs)
            fa = failure_analysis(by[args.ours], n_examples=args.n_failures)
            (args.results_dir / "failure_analysis.md").write_text(fa, encoding="utf-8")
            LOGGER.info("wrote %s", args.results_dir / "failure_analysis.md")

    if args.ablation:
        abl = [load_run(s) for s in args.ablation]
        chart_ablation(abl, charts_dir)
        (args.results_dir / "ablation_table.md").write_text(ablation_table(abl), encoding="utf-8")
        LOGGER.info("wrote %s", args.results_dir / "ablation_table.md")

    if args.robustness:
        ours = json.loads(args.robustness.read_text(encoding="utf-8"))
        base = json.loads(args.robustness_baseline.read_text(encoding="utf-8")) if args.robustness_baseline else None
        chart_robustness(ours, base, charts_dir)
        LOGGER.info("wrote robustness chart")

    return 0


if __name__ == "__main__":
    sys.exit(main())
