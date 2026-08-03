"""Metric tests.

The important property being pinned here is that the metrics cannot be gamed by
the degenerate strategies a weak model actually falls into: always predict null,
never produce parseable output, or emit everything. Each of those has a test
asserting it scores badly.
"""

from __future__ import annotations

from data.schema import JobPosting, Location, Salary, flatten, parse_prediction, to_target_json
from eval.metrics import (
    Counts,
    aggregate,
    bootstrap_ci,
    paired_bootstrap_pvalue,
    score_example,
    values_equal,
)


def gold_obj() -> JobPosting:
    return JobPosting(
        job_title="Backend Engineer",
        company_name="Acme",
        location=Location(city="Austin", region="TX", country="United States", remote_policy="hybrid"),
        employment_type="full_time",
        seniority_level="senior",
        education_requirement="bachelor",
        salary=Salary(min_amount=120000, max_amount=150000, currency="USD", period="yearly"),
        years_experience_min=5,
        required_skills=["python", "sql"],
        preferred_skills=["go"],
        benefits=["401k"],
    )


SOURCE = ("Acme is hiring a Backend Engineer in Austin, TX, United States. Hybrid. "
          "Full-time, senior level. $120,000 - $150,000 per year. 5 years experience. "
          "Bachelor's degree. Python and SQL required. Go preferred. 401k offered.")


class TestComparators:
    def test_string_comparison_ignores_case_and_punctuation(self):
        assert values_equal("job_title", "Backend Engineer", "backend engineer,")

    def test_categorical_is_exact(self):
        assert not values_equal("employment_type", "full_time", "fulltime")

    def test_salary_has_a_tolerance(self):
        assert values_equal("salary.min_amount", 120000, 121000)      # within 2%
        assert not values_equal("salary.min_amount", 120000, 130000)  # 8% out

    def test_years_experience_is_exact(self):
        # No tolerance here on purpose: off-by-one on "3-5 years" is a reasoning
        # error we want counted, not absorbed.
        assert not values_equal("years_experience_min", 5, 4)

    def test_sets_are_order_insensitive(self):
        assert values_equal("required_skills", ["python", "sql"], ["sql", "python"])

    def test_null_matches_only_null(self):
        assert values_equal("company_name", None, None)
        assert not values_equal("company_name", None, "Acme")


class TestCounts:
    def test_wrong_value_counts_as_both_fp_and_fn(self):
        c = Counts()
        c.update("company_name", "Acme", "Globex")
        assert (c.tp, c.fp, c.fn) == (0, 1, 1)

    def test_over_emission_is_a_false_positive(self):
        c = Counts()
        c.update("company_name", None, "Globex")
        assert (c.fp, c.fn) == (1, 0)

    def test_sets_get_partial_credit(self):
        c = Counts()
        c.update("required_skills", ["python", "sql", "aws"], ["python", "sql", "go"])
        assert (c.tp, c.fp, c.fn) == (2, 1, 1)


class TestScoring:
    def test_perfect_prediction(self):
        g = to_target_json(gold_obj())
        res = score_example("p1", SOURCE, g, g)
        assert res.exact_match and res.strict_compliant and res.lenient_compliant
        assert res.ungrounded == []

    def test_fenced_output_is_lenient_but_not_strict(self):
        g = to_target_json(gold_obj())
        res = score_example("p1", SOURCE, g, f"```json\n{g}\n```")
        assert res.lenient_compliant and not res.strict_compliant

    def test_unparseable_output_is_scored_not_skipped(self):
        g = to_target_json(gold_obj())
        res = score_example("p1", SOURCE, g, "sorry, I can't")
        assert not res.lenient_compliant
        # Every non-null gold field must be counted wrong, otherwise a model that
        # refuses to answer would outscore one that tries.
        assert not any(v for k, v in res.per_field_correct.items()
                       if flatten(gold_obj())[k] is not None)

    def test_hallucinated_value_is_flagged(self):
        obj = gold_obj().model_copy(update={"company_name": "Globex"})
        res = score_example("p1", SOURCE, to_target_json(gold_obj()), to_target_json(obj))
        assert "company_name" in res.ungrounded


class TestAggregate:
    def _run(self, preds_json: list[str]) -> dict:
        gold = gold_obj()
        gj = to_target_json(gold)
        results, golds, preds = [], {}, {}
        for i, pj in enumerate(preds_json):
            pid = f"p{i}"
            results.append(score_example(pid, SOURCE, gj, pj))
            golds[pid], _ = parse_prediction(gj)
            preds[pid], _ = parse_prediction(pj)
        return aggregate(results, golds, preds)

    def test_perfect_run(self):
        m = self._run([to_target_json(gold_obj())] * 5)
        assert m["exact_match"] == 1.0
        assert m["field_f1_micro"] == 1.0
        assert m["hallucination_rate"] == 0.0

    def test_always_null_strategy_scores_zero_f1(self):
        # The degenerate strategy this metric design exists to defeat.
        null_pred = to_target_json(JobPosting(job_title="Backend Engineer"))
        m = self._run([null_pred] * 5)
        assert m["field_f1_micro"] < 0.2
        assert m["null_recall"] == 1.0  # ...while still looking perfect on nulls

    def test_refusal_scores_zero_compliance(self):
        m = self._run(["I cannot help"] * 5)
        assert m["schema_compliance_lenient"] == 0.0
        assert m["field_f1_micro"] == 0.0


class TestBootstrap:
    def _fixture(self, n: int = 30):
        gold = gold_obj()
        gj = to_target_json(gold)
        wrong = to_target_json(gold.model_copy(update={"company_name": "Globex"}))
        results, golds, preds = [], {}, {}
        for i in range(n):
            pid = f"p{i}"
            pj = gj if i % 2 == 0 else wrong
            results.append(score_example(pid, SOURCE, gj, pj))
            golds[pid], _ = parse_prediction(gj)
            preds[pid], _ = parse_prediction(pj)
        return results, golds, preds

    def test_ci_brackets_the_point_estimate(self):
        results, golds, preds = self._fixture()
        point = aggregate(results, golds, preds)
        ci = bootstrap_ci(results, golds, preds, n_boot=200)
        lo, hi = ci["field_f1_micro"]
        assert lo <= point["field_f1_micro"] <= hi

    def test_identical_models_are_not_significant(self):
        results, golds, preds = self._fixture()
        p = paired_bootstrap_pvalue(results, results, golds, preds, preds, n_boot=200)
        assert p > 0.5
