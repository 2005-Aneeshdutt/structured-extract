"""Tests for the reporting layer.

These exist because the reporting scripts run *last* -- after five days of
labeling and hours of GPU time -- so a crash there is maximally expensive. The
`load_run` case below is a real bug that shipped: ablation labels contain `=`
("r=8=path.json") and splitting on the first `=` produced a FileNotFoundError
citing a path the user never typed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.schema import JobPosting, Location, Salary, to_target_json
from eval.compare_models import gap_closed, load_run
from eval.generate_report import FAILURE_CATEGORIES, categorize, failure_analysis


def _payload(tmp: Path, name: str, exact: bool = False) -> Path:
    gold = JobPosting(
        job_title="Backend Engineer",
        company_name="Acme",
        location=Location(city="Austin", region="TX"),
        salary=Salary(min_amount=120000, max_amount=150000, currency="USD", period="yearly"),
        years_experience_min=5,
        required_skills=["python", "sql"],
    )
    gj = to_target_json(gold)
    pred = gj if exact else to_target_json(gold.model_copy(update={"company_name": "Globex"}))
    payload = {
        "backend": name,
        "n": 1,
        "metrics": {"schema_compliance_lenient": 1.0, "field_f1_micro": 0.5,
                    "exact_match": 1.0 if exact else 0.0, "hallucination_rate": 0.0,
                    "per_field": {}, "per_tier": {}},
        "ci95": {},
        "per_example": [{
            "posting_id": "p1", "raw_output": pred, "gold_json": gj,
            "source_excerpt": "Acme is hiring a Backend Engineer in Austin, TX. "
                              "$120,000 - $150,000 per year. 5 years. Python and SQL.",
            "latency_s": 1.0, "parse_error": None, "exact_match": exact, "ungrounded": [],
        }],
        "results": [],
        "config": {},
    }
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


class TestLoadRun:
    def test_label_containing_equals_is_parsed_correctly(self, tmp_path: Path):
        """`r=8=path.json` -> label 'r=8', not label 'r' and a bogus path."""
        p = _payload(tmp_path, "run")
        label, payload = load_run(f"r=8={p}")
        assert label == "r=8"
        assert payload["backend"] == "run"

    def test_plain_label(self, tmp_path: Path):
        p = _payload(tmp_path, "run")
        label, _ = load_run(f"Base 0-shot={p}")
        assert label == "Base 0-shot"

    def test_missing_file_names_the_spec_it_came_from(self, tmp_path: Path):
        with pytest.raises(SystemExit, match="not found"):
            load_run(f"r=8={tmp_path / 'nope.json'}")

    def test_missing_equals_is_rejected(self):
        with pytest.raises(SystemExit, match="LABEL=PATH"):
            load_run("just-a-path.json")

    def test_empty_label_is_rejected(self, tmp_path: Path):
        p = _payload(tmp_path, "run")
        with pytest.raises(SystemExit, match="non-empty label"):
            load_run(f"={p}")


class TestGapClosed:
    def test_half_the_gap(self):
        assert gap_closed(0.4, 0.6, 0.8) == pytest.approx(0.5)

    def test_exceeding_the_ceiling_is_reported_not_clipped(self):
        assert gap_closed(0.4, 0.9, 0.8) > 1.0

    def test_degenerate_gap_returns_none(self):
        # "closed 400% of the gap" is what a near-zero denominator produces, and
        # it is exactly the number that gets a project dismissed.
        assert gap_closed(0.8, 0.9, 0.8) is None
        assert gap_closed(0.8, 0.9, 0.7) is None

    def test_lower_is_better_direction(self):
        # hallucination: baseline 20%, ours 10%, ceiling 0% -> half the gap
        assert gap_closed(0.2, 0.1, 0.0, lower_better=True) == pytest.approx(0.5)


class TestFailureCategorization:
    def _row(self, raw: str, gold: str, **kw):
        return {"posting_id": "p1", "raw_output": raw, "gold_json": gold, **kw}

    def test_unparseable_short_circuits(self):
        assert categorize(self._row("nonsense", "{}", parse_error="no_json")) == ["unparseable"]

    def test_over_emission_detected(self):
        gold = to_target_json(JobPosting(job_title="Cook"))
        pred = to_target_json(JobPosting(job_title="Cook", company_name="Invented Inc"))
        assert "over_emission" in categorize(self._row(pred, gold))

    def test_missed_extraction_detected(self):
        gold = to_target_json(JobPosting(job_title="Cook", company_name="Acme"))
        pred = to_target_json(JobPosting(job_title="Cook"))
        assert "missed_extraction" in categorize(self._row(pred, gold))

    def test_salary_error_is_its_own_category(self):
        gold = to_target_json(JobPosting(job_title="Cook", salary=Salary(min_amount=100000)))
        pred = to_target_json(JobPosting(job_title="Cook", salary=Salary(min_amount=10000)))
        assert "salary_misparse" in categorize(self._row(pred, gold))

    def test_every_category_has_a_description(self):
        # Guards against adding a category to categorize() without documenting it
        # in the taxonomy table the README points readers at.
        gold = to_target_json(JobPosting(job_title="Cook"))
        pred = to_target_json(JobPosting(job_title="Cook", company_name="X"))
        for cat in categorize(self._row(pred, gold)):
            assert cat in FAILURE_CATEGORIES


class TestFailureAnalysisReport:
    def test_renders_without_a_model(self, tmp_path: Path):
        _, payload = load_run(f"run={_payload(tmp_path, 'run')}")
        md = failure_analysis(payload, n_examples=3)
        assert "# Failure analysis" in md
        assert "Failure taxonomy" in md
        assert "company_name" in md  # the injected difference shows up in the diff table
