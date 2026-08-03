"""Adversarial / edge-case robustness evaluation.

    python -m eval.robustness_test --backend hf --adapter outputs/.../adapter \
        --out results/raw_predictions/robustness_finetuned.json

The idea, and why it is built this way
--------------------------------------
Robustness suites usually degenerate into "we eyeballed some weird inputs". That
is not measurable. Instead every perturbation here is a **deterministic function
of a test example that also determines the correct label**:

* *label-invariant* perturbations change only surface form (whitespace, case,
  HTML residue, appended boilerplate). The gold label is unchanged by
  construction, so any metric drop is a genuine brittleness -- there is nothing
  to argue about.
* *label-transforming* perturbations delete information (the salary sentence, the
  deadline sentence). The new gold is computed by applying the same deletion to
  the label. These are the interesting ones: they measure whether the model
  **stops asserting a value once its evidence is removed**, which is exactly the
  hallucination behavior that distinguishes a fine-tuned extractor from a base
  model that pattern-matches "job posting -> emit a salary".

Because both the perturbed input and the perturbed gold are derived
programmatically, this whole suite costs zero API calls and zero human labeling.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from data.corpus import SALARY_IN_TEXT_RE
from data.schema import JobPosting, Salary, flatten, parse_prediction, to_target_json
from eval.metrics import aggregate, score_example
from eval.run_eval import build_backend, load_fewshot, load_split

LOGGER = logging.getLogger("robustness")

Perturbation = Callable[[str, JobPosting, random.Random], tuple[str, JobPosting]]

_EEO_BOILERPLATE = (
    "\n\nEqual Opportunity Employer Statement. We are proud to be an equal opportunity "
    "workplace and are an affirmative action employer. We are committed to equal employment "
    "opportunity regardless of race, color, ancestry, religion, sex, national origin, sexual "
    "orientation, age, citizenship, marital status, disability, gender identity or Veteran "
    "status. We also consider qualified applicants regardless of criminal histories, "
    "consistent with legal requirements. If you have a disability or special need that "
    "requires accommodation, please let us know. This job description is not designed to "
    "cover or contain a comprehensive listing of activities, duties or responsibilities "
    "that are required of the employee. Duties, responsibilities and activities may change "
    "at any time with or without notice. Applicants must be authorized to work for any "
    "employer in the country of employment. We are unable to sponsor or take over "
    "sponsorship of an employment visa at this time."
)


# ---------------------------------------------------------------------------
# Label-invariant perturbations
# ---------------------------------------------------------------------------


def p_identity(text: str, gold: JobPosting, rng: random.Random) -> tuple[str, JobPosting]:
    """Control arm. Its score should match the main test-set score.

    Included so the robustness table has an internal consistency check: if
    `identity` disagrees with the headline number, the harness is broken, not the
    model.
    """
    return text, gold


def p_whitespace_chaos(text: str, gold: JobPosting, rng: random.Random) -> tuple[str, JobPosting]:
    """Random extra blank lines, tabs and doubled spaces -- what a bad PDF-to-text
    or copy-paste from a browser produces."""
    out = []
    for line in text.split("\n"):
        if rng.random() < 0.3:
            line = line.replace(" ", "  ", rng.randint(1, 5))
        out.append(line)
        if rng.random() < 0.25:
            out.append("")
        if rng.random() < 0.1:
            out.append("\t")
    return "\n".join(out), gold


def p_shout_case(text: str, gold: JobPosting, rng: random.Random) -> tuple[str, JobPosting]:
    """ALL CAPS. Label-invariant because every string comparator in metrics.py
    casefolds, and every enum/list value is canonicalized lowercase."""
    return text.upper(), gold


def p_html_residue(text: str, gold: JobPosting, rng: random.Random) -> tuple[str, JobPosting]:
    """Leftover tags and entities -- the single most common real-world defect in
    a scraped posting that skipped a sanitizer."""
    tags = ["<p>", "</p>", "<br/>", "<div>", "</div>", "<span>", "</span>", "<li>", "</li>", "<strong>"]
    lines = text.split("\n")
    out = []
    for line in lines:
        if rng.random() < 0.5:
            line = rng.choice(tags) + line + rng.choice(tags)
        line = line.replace(" & ", " &amp; ").replace("'", "&#39;")
        if rng.random() < 0.3:
            line = line.replace(" ", "&nbsp;", 2)
        out.append(line)
    return "\n".join(out), gold


def p_mojibake(text: str, gold: JobPosting, rng: random.Random) -> tuple[str, JobPosting]:
    """Re-introduce the UTF-8-decoded-as-latin1 damage that ftfy repaired during
    data prep. Tests whether the model depends on our cleaning step -- a
    deployment will not always run it."""
    subs = {"'": "â€™", '"': "â€œ", "-": "â€“", "·": "Â·", "é": "Ã©"}
    for k, v in subs.items():
        text = text.replace(k, v)
    return text, gold


def p_boilerplate_flood(text: str, gold: JobPosting, rng: random.Random) -> tuple[str, JobPosting]:
    """Append ~1.2 KB of EEO legalese.

    Appended at the END on purpose: head-truncation means the model still sees
    all the label-bearing content, so this isolates 'does irrelevant volume
    distract it' from 'did we truncate the answer away'.
    """
    return text + _EEO_BOILERPLATE, gold


def p_typos(text: str, gold: JobPosting, rng: random.Random) -> tuple[str, JobPosting]:
    """Character-level noise, applied ONLY to tokens that appear in no gold value.

    The guard is what makes this label-invariant. Corrupting a token that IS the
    answer would change the correct label, and the run would silently be
    measuring label noise instead of robustness.
    """
    protected: set[str] = set()
    for v in flatten(gold).values():
        if isinstance(v, str):
            protected |= {w.lower() for w in v.split()}
        elif isinstance(v, list):
            for item in v:
                protected |= {w.lower() for w in str(item).split()}
        elif isinstance(v, (int, float)):
            protected |= {str(v), str(int(v)), f"{int(v):,}"}

    words = text.split(" ")
    for i, w in enumerate(words):
        if len(w) < 5 or w.lower().strip(".,;:") in protected or rng.random() > 0.06:
            continue
        j = rng.randrange(1, len(w) - 1)
        words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]  # adjacent transposition
    return " ".join(words), gold


# ---------------------------------------------------------------------------
# Label-transforming perturbations -- the hallucination probes
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?\n])\s+")


def p_strip_salary(text: str, gold: JobPosting, rng: random.Random) -> tuple[str, JobPosting]:
    """Delete every sentence containing a currency amount; gold salary -> all null.

    THE key robustness test. A model that has learned the task emits
    `"salary": {"min_amount": null, ...}`. A model that has learned the *genre*
    keeps emitting a plausible band because job postings usually have one. The
    difference shows up as over_emission_rate on this arm and nowhere else.
    """
    kept = [s for s in _SENT_SPLIT.split(text) if not SALARY_IN_TEXT_RE.search(s)]
    new_gold = gold.model_copy(update={"salary": Salary()})
    return " ".join(kept), new_gold


def p_strip_deadline(text: str, gold: JobPosting, rng: random.Random) -> tuple[str, JobPosting]:
    """Delete sentences mentioning a deadline; gold application_deadline -> null."""
    cue = re.compile(r"\b(?:deadline|apply by|closing date|applications? close|submit by)\b", re.IGNORECASE)
    kept = [s for s in _SENT_SPLIT.split(text) if not cue.search(s)]
    return " ".join(kept), gold.model_copy(update={"application_deadline": None})


def p_truncate_tail(text: str, gold: JobPosting, rng: random.Random) -> tuple[str, JobPosting]:
    """Keep only the first 40% of the posting.

    NOT label-invariant in general, so it is scored on a restricted field set:
    only tier-1/tier-2 fields (title, company, location), which live in the
    opening paragraphs. Reported separately in the table for that reason.
    """
    cut = max(400, int(len(text) * 0.4))
    return text[:cut], gold


PERTURBATIONS: dict[str, tuple[Perturbation, str]] = {
    "identity":            (p_identity, "control -- should match the headline test score"),
    "whitespace_chaos":    (p_whitespace_chaos, "label-invariant: erratic spacing/newlines"),
    "shout_case":          (p_shout_case, "label-invariant: ALL CAPS"),
    "html_residue":        (p_html_residue, "label-invariant: leftover tags and entities"),
    "mojibake":            (p_mojibake, "label-invariant: broken UTF-8 decoding"),
    "boilerplate_flood":   (p_boilerplate_flood, "label-invariant: +1.2KB of EEO legalese"),
    "typos":               (p_typos, "label-invariant: transpositions, gold tokens protected"),
    "strip_salary":        (p_strip_salary, "label-transforming: salary evidence removed -> gold null"),
    "strip_deadline":      (p_strip_deadline, "label-transforming: deadline evidence removed -> gold null"),
    "truncate_tail":       (p_truncate_tail, "distribution shift: first 40% only (tier 1-2 fields only)"),
}

#: truncate_tail cannot preserve labels for fields that live in the deleted tail,
#: so it is scored on the fields that provably survive.
RESTRICTED_FIELDS = {"truncate_tail": ("job_title", "company_name", "location.city",
                                       "location.region", "location.country")}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_suite(examples: list[dict[str, Any]], names: list[str], n_per: int, seed: int
                ) -> dict[str, list[dict[str, Any]]]:
    """Materialize the perturbed evaluation sets.

    The same `n_per` source examples are used for every perturbation, so the
    arms are paired: a difference between `identity` and `mojibake` is the
    perturbation, not a different sample of postings.
    """
    rng = random.Random(seed)
    base = examples[:n_per]
    suite: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        fn, _desc = PERTURBATIONS[name]
        rows = []
        for ex in base:
            gold, err = parse_prediction(ex["target_json"])
            if gold is None:
                LOGGER.warning("skipping %s: gold does not validate (%s)", ex["posting_id"], err)
                continue
            new_text, new_gold = fn(ex["source_text"], gold, random.Random(rng.randrange(1 << 30)))
            rows.append({
                "posting_id": f"{ex['posting_id']}::{name}",
                "source_text": new_text,
                "target_json": to_target_json(new_gold),
            })
        suite[name] = rows
    return suite


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["hf", "gguf", "gemini"], required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--n-gpu-layers", type=int, default=0)
    ap.add_argument("--gemini-model", default="gemini-2.0-flash")
    ap.add_argument("--unconstrained-gemini", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--few-shot", type=int, default=0)
    ap.add_argument("--n-per-perturbation", type=int, default=100,
                    help="100 x 10 perturbations = 1000 generations; ~25 min on a 2050 in 4-bit")
    ap.add_argument("--perturbations", nargs="*", default=list(PERTURBATIONS))
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    examples = load_split(args.data_dir, args.split)
    few_shot = load_fewshot(args.data_dir, args.few_shot)
    suite = build_suite(examples, args.perturbations, args.n_per_perturbation, args.seed)
    backend = build_backend(args)

    report: dict[str, Any] = {"backend": backend.name, "n_per_perturbation": args.n_per_perturbation,
                              "arms": {}}
    for name, rows in suite.items():
        results, preds, golds = [], {}, {}
        for ex in rows:
            try:
                out = backend.generate(ex["source_text"], few_shot)
            except Exception as e:
                LOGGER.warning("generation failed: %s", e)
                out = ""
            results.append(score_example(ex["posting_id"], ex["source_text"], ex["target_json"], out))
            preds[ex["posting_id"]], _ = parse_prediction(out)
            golds[ex["posting_id"]], _ = parse_prediction(ex["target_json"])
        m = aggregate(results, golds, preds)
        if name in RESTRICTED_FIELDS:
            keep = RESTRICTED_FIELDS[name]
            m["field_f1_macro_restricted"] = sum(
                m["per_field"][f]["f1"] for f in keep) / len(keep)
            m["restricted_to"] = list(keep)
        report["arms"][name] = {"description": PERTURBATIONS[name][1], "metrics": m}
        LOGGER.info("%-20s compliance %.1f%% | F1 %.3f | over-emission %.1f%%",
                    name, 100 * m["schema_compliance_lenient"], m["field_f1_micro"],
                    100 * m["over_emission_rate"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
