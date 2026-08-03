"""Metric definitions. Every number in the README is computed here.

The design principle throughout: **a metric a reviewer cannot argue with**. That
means every comparator is exact or has an explicitly stated, justified tolerance;
no fuzzy string similarity thresholds, no "close enough" heuristics that could be
tuned until the numbers look good.

Metric set
----------
schema_compliance_strict   raw completion parses as JSON *and* validates
schema_compliance_lenient  parses after stripping markdown fences / prose
exact_match                every one of the 18 leaf fields is correct
field P / R / F1           per field, then micro- and macro-averaged
hallucination_rate         share of emitted values not supported by the source
over_emission_rate         share of emitted values where the gold is null
null_recall                share of gold-null fields the model correctly left null

Why precision/recall rather than plain accuracy
-----------------------------------------------
Most fields in this schema are null most of the time (see
results/dataset_stats.md). Accuracy on `application_deadline` is ~97% for a model
that always emits null -- a number that looks excellent and means nothing. P/R/F1
with nulls treated as "no prediction" makes the majority-class shortcut score
zero, which is the honest outcome.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from data.schema import (
    FIELD_SPECS,
    TIER_NAMES,
    JobPosting,
    flatten,
    parse_prediction,
    ungrounded_fields,
)

#: Relative tolerance for salary amounts. 2% absorbs the genuine ambiguity in
#: "around $120k" / "$119,500-$120,000" without excusing a real misparse (a
#: 120000 vs 12000 error is 90% off, a 120000 vs 150000 error is 25% off -- both
#: are caught). years_experience_min uses EXACT match: "3-5 years" has one
#: correct answer and tolerance there would hide off-by-one reasoning errors.
SALARY_REL_TOL = 0.02

_PUNCT_RE = re.compile(r"[^\w\s]")


def _norm_string(v: str) -> str:
    """Casefold, strip punctuation, collapse whitespace.

    This is the *only* normalization applied at scoring time, and it is applied
    identically to gold and prediction. It exists so that "Austin," and "austin"
    are not counted as a miss -- a difference no consumer of the JSON would care
    about. Anything beyond this (abbreviation expansion, fuzzy distance) would be
    scoring generosity rather than correctness.
    """
    return " ".join(_PUNCT_RE.sub(" ", v.lower()).split())


def values_equal(field_name: str, gold: Any, pred: Any) -> bool:
    """Type-aware equality, dispatched on the comparator in FIELD_SPECS."""
    comparator = FIELD_SPECS[field_name]["comparator"]
    if gold is None or pred is None:
        return gold is None and pred is None
    if comparator == "string":
        return _norm_string(str(gold)) == _norm_string(str(pred))
    if comparator in ("categorical", "date"):
        return str(gold) == str(pred)
    if comparator == "numeric":
        try:
            g, p = float(gold), float(pred)
        except (TypeError, ValueError):
            return False
        if field_name.startswith("salary."):
            return abs(g - p) <= SALARY_REL_TOL * max(abs(g), 1.0)
        return g == p
    if comparator == "set":
        return set(gold or []) == set(pred or [])
    raise ValueError(f"unknown comparator for {field_name}")


@dataclass
class Counts:
    """TP/FP/FN/TN for one field, accumulated across the test set.

    Null handling, stated explicitly because it drives every headline number:

      gold non-null, pred non-null, equal      -> TP
      gold non-null, pred non-null, unequal    -> FP *and* FN (a wrong value is
                                                  both a bad emission and a
                                                  missed extraction)
      gold non-null, pred null                 -> FN
      gold null,     pred non-null             -> FP  (this is over-emission)
      gold null,     pred null                 -> TN

    Set-valued fields accumulate element-level counts instead, so partially
    correct skill lists get partial credit -- an all-or-nothing rule there would
    report near-zero F1 for a genuinely useful model.
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def support(self) -> int:
        return self.tp + self.fn

    def update(self, field_name: str, gold: Any, pred: Any) -> None:
        if FIELD_SPECS[field_name]["comparator"] == "set":
            g, p = set(gold or []), set(pred or [])
            self.tp += len(g & p)
            self.fp += len(p - g)
            self.fn += len(g - p)
            if not g and not p:
                self.tn += 1
            return
        g_null, p_null = gold is None, pred is None
        if g_null and p_null:
            self.tn += 1
        elif g_null and not p_null:
            self.fp += 1
        elif not g_null and p_null:
            self.fn += 1
        elif values_equal(field_name, gold, pred):
            self.tp += 1
        else:
            self.fp += 1
            self.fn += 1


@dataclass
class ExampleResult:
    """Per-example scoring record. Kept so bootstrap resampling is cheap and so
    failure analysis can filter on any of these flags without re-running the
    model."""

    posting_id: str
    raw_output: str
    parse_error: str | None
    strict_compliant: bool
    lenient_compliant: bool
    exact_match: bool
    per_field_correct: dict[str, bool] = field(default_factory=dict)
    ungrounded: list[str] = field(default_factory=list)
    n_emitted: int = 0
    n_over_emitted: int = 0
    n_gold_null: int = 0
    n_correct_null: int = 0
    latency_s: float = 0.0


def score_example(posting_id: str, source_text: str, gold_json: str, raw_output: str,
                  latency_s: float = 0.0) -> ExampleResult:
    """Score one completion against one gold label."""
    # Strict = the completion is a bare JSON object with nothing around it.
    # Lenient = it survives fence-stripping and brace-matching. Reporting both is
    # what lets the README say how much of the base model's failure is cosmetic
    # (markdown fences) versus structural (truncated / invalid objects).
    strict = False
    try:
        json.loads(raw_output.strip())
        strict = True
    except (json.JSONDecodeError, ValueError):
        strict = False

    pred, err = parse_prediction(raw_output)
    strict = strict and pred is not None

    gold, gold_err = parse_prediction(gold_json)
    if gold is None:
        raise ValueError(f"gold label for {posting_id} does not validate: {gold_err}")

    res = ExampleResult(
        posting_id=posting_id,
        raw_output=raw_output,
        parse_error=err,
        strict_compliant=strict,
        lenient_compliant=pred is not None,
        exact_match=False,
        latency_s=latency_s,
    )
    gf = flatten(gold)
    if pred is None:
        # A non-parsing completion is scored as "predicted nothing" -- every gold
        # non-null becomes a false negative. It is NOT skipped: dropping
        # unparseable outputs would let a model that fails half the time post a
        # better F1 than one that always answers.
        res.per_field_correct = {k: (gf[k] is None) for k in FIELD_SPECS}
        res.n_gold_null = sum(1 for v in gf.values() if v is None)
        res.n_correct_null = res.n_gold_null
        return res

    pf = flatten(pred)
    res.exact_match = all(values_equal(k, gf[k], pf[k]) for k in FIELD_SPECS)
    for k in FIELD_SPECS:
        res.per_field_correct[k] = values_equal(k, gf[k], pf[k])
        emitted = pf[k] not in (None, [])
        if emitted:
            res.n_emitted += 1
            if gf[k] in (None, []):
                res.n_over_emitted += 1
        if gf[k] is None:
            res.n_gold_null += 1
            if pf[k] is None:
                res.n_correct_null += 1
    res.ungrounded = ungrounded_fields(pred, source_text)
    return res


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(results: Sequence[ExampleResult], golds: dict[str, JobPosting],
              preds: dict[str, JobPosting | None]) -> dict[str, Any]:
    """Roll per-example results into the reported metric set.

    `golds`/`preds` are passed separately (rather than recomputed) so the
    bootstrap can resample examples without re-parsing thousands of JSON strings.
    """
    n = len(results)
    if n == 0:
        return {}

    counts: dict[str, Counts] = {k: Counts() for k in FIELD_SPECS}
    for r in results:
        gf = flatten(golds[r.posting_id])
        pred = preds.get(r.posting_id)
        pf = flatten(pred) if pred is not None else dict.fromkeys(FIELD_SPECS)
        for k in FIELD_SPECS:
            counts[k].update(k, gf[k], pf[k])

    micro = Counts()
    for c in counts.values():
        micro.tp += c.tp
        micro.fp += c.fp
        micro.fn += c.fn
        micro.tn += c.tn

    per_field = {
        k: {"precision": c.precision, "recall": c.recall, "f1": c.f1,
            "support": c.support, "tier": FIELD_SPECS[k]["tier"]}
        for k, c in counts.items()
    }
    per_tier = {}
    for tier, name in TIER_NAMES.items():
        fs = [k for k in FIELD_SPECS if FIELD_SPECS[k]["tier"] == tier]
        if fs:
            per_tier[f"tier{tier}_{name.replace(' ', '_')}"] = sum(per_field[k]["f1"] for k in fs) / len(fs)

    n_emitted = sum(r.n_emitted for r in results)
    n_ungrounded = sum(len(r.ungrounded) for r in results)
    n_gold_null = sum(r.n_gold_null for r in results)

    return {
        "n": n,
        "schema_compliance_strict": sum(r.strict_compliant for r in results) / n,
        "schema_compliance_lenient": sum(r.lenient_compliant for r in results) / n,
        "exact_match": sum(r.exact_match for r in results) / n,
        "field_f1_micro": micro.f1,
        "field_precision_micro": micro.precision,
        "field_recall_micro": micro.recall,
        "field_f1_macro": sum(v["f1"] for v in per_field.values()) / len(per_field),
        # Denominator is emitted values, not examples: "of everything the model
        # asserted, what fraction was not supported by the text". That is the
        # number a user of the extractor actually cares about.
        "hallucination_rate": n_ungrounded / max(n_emitted, 1),
        "over_emission_rate": sum(r.n_over_emitted for r in results) / max(n_emitted, 1),
        "null_recall": sum(r.n_correct_null for r in results) / max(n_gold_null, 1),
        "mean_latency_s": sum(r.latency_s for r in results) / n,
        "per_field": per_field,
        "per_tier": per_tier,
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

HEADLINE_METRICS = (
    "schema_compliance_strict",
    "schema_compliance_lenient",
    "exact_match",
    "field_f1_micro",
    "field_f1_macro",
    "hallucination_rate",
    "over_emission_rate",
    "null_recall",
)


def bootstrap_ci(
    results: Sequence[ExampleResult],
    golds: dict[str, JobPosting],
    preds: dict[str, JobPosting | None],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 17,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap 95% CI over the headline metrics.

    Resampling is at the **example** level, with replacement, n out of n. This is
    the part people get wrong: resampling at the field level would treat the 18
    fields of one posting as 18 independent observations, which they are not --
    a posting the model fails to parse fails all 18 at once. Field-level
    resampling would therefore report intervals roughly sqrt(18) times too
    narrow, and a 2-point difference between models would look significant when
    it is not.

    500 test examples gives roughly +/- 3-4 pp on a rate near 0.5, which is the
    resolution at which the model comparison in the README should be read. n_boot
    = 1000 is enough for a percentile interval at alpha=0.05; beyond that the
    Monte Carlo error is well under the sampling error it estimates.
    """
    rng = random.Random(seed)
    idx = list(range(len(results)))
    samples: dict[str, list[float]] = {m: [] for m in HEADLINE_METRICS}

    for _ in range(n_boot):
        draw = [results[rng.choice(idx)] for _ in idx]
        agg = aggregate(draw, golds, preds)
        for m in HEADLINE_METRICS:
            samples[m].append(agg.get(m, math.nan))

    out: dict[str, tuple[float, float]] = {}
    lo_i = int(alpha / 2 * n_boot)
    hi_i = int((1 - alpha / 2) * n_boot) - 1
    for m, vals in samples.items():
        vals.sort()
        out[m] = (vals[max(0, lo_i)], vals[min(len(vals) - 1, hi_i)])
    return out


def paired_bootstrap_pvalue(
    results_a: Sequence[ExampleResult],
    results_b: Sequence[ExampleResult],
    golds: dict[str, JobPosting],
    preds_a: dict[str, JobPosting | None],
    preds_b: dict[str, JobPosting | None],
    metric: str = "field_f1_micro",
    *,
    n_boot: int = 1000,
    seed: int = 17,
) -> float:
    """Two-sided paired bootstrap p-value for 'model A differs from model B'.

    Paired, because both models are scored on the *same* postings -- an easy
    posting is easy for both. Pairing removes that shared variance and is
    strictly more powerful than comparing two independent CIs, which is why
    "the intervals overlap, therefore no difference" is a mistake this function
    exists to avoid making.
    """
    rng = random.Random(seed)
    by_id_a = {r.posting_id: r for r in results_a}
    by_id_b = {r.posting_id: r for r in results_b}
    ids = [i for i in by_id_a if i in by_id_b]
    if not ids:
        return float("nan")

    observed = (aggregate([by_id_a[i] for i in ids], golds, preds_a).get(metric, 0.0)
                - aggregate([by_id_b[i] for i in ids], golds, preds_b).get(metric, 0.0))

    n_extreme = 0
    for _ in range(n_boot):
        draw = [rng.choice(ids) for _ in ids]
        d = (aggregate([by_id_a[i] for i in draw], golds, preds_a).get(metric, 0.0)
             - aggregate([by_id_b[i] for i in draw], golds, preds_b).get(metric, 0.0))
        # Centered on the observed difference -> tests H0: no difference.
        if abs(d - observed) >= abs(observed):
            n_extreme += 1
    return n_extreme / n_boot
