"""Tests for the perturbation suite.

The suite's entire validity rests on one property: a *label-invariant*
perturbation must not change the gold label, and a *label-transforming* one must
change it in exactly the stated way. If that breaks, the robustness table stops
measuring robustness and starts measuring label noise -- silently. Hence these
tests.
"""

from __future__ import annotations

import random

from data.schema import JobPosting, Location, Salary, flatten, to_target_json
from eval.robustness_test import (
    PERTURBATIONS,
    RESTRICTED_FIELDS,
    build_suite,
    p_strip_deadline,
    p_strip_salary,
    p_typos,
)

RNG = random.Random(0)

SOURCE = (
    "Senior Backend Engineer at Acme in Austin, TX. Full-time, hybrid.\n"
    "Salary: $120,000 - $150,000 per year.\n"
    "Requires 5 years of experience with Python and SQL. Bachelor's degree required.\n"
    "Application deadline: 2025-03-14. Benefits include 401k."
)


def gold() -> JobPosting:
    return JobPosting(
        job_title="Senior Backend Engineer",
        company_name="Acme",
        location=Location(city="Austin", region="TX", remote_policy="hybrid"),
        employment_type="full_time",
        seniority_level="senior",
        education_requirement="bachelor",
        salary=Salary(min_amount=120000, max_amount=150000, currency="USD", period="yearly"),
        years_experience_min=5,
        required_skills=["python", "sql"],
        benefits=["401k"],
        application_deadline="2025-03-14",
    )


LABEL_INVARIANT = ["identity", "whitespace_chaos", "shout_case", "html_residue",
                   "mojibake", "boilerplate_flood", "typos"]


class TestLabelInvariance:
    def test_invariant_perturbations_never_change_the_gold(self):
        g = gold()
        before = to_target_json(g)
        for name in LABEL_INVARIANT:
            fn, _desc = PERTURBATIONS[name]
            text, new_gold = fn(SOURCE, g, random.Random(1))
            assert to_target_json(new_gold) == before, f"{name} mutated the label"
            assert text, f"{name} produced empty text"

    def test_invariant_perturbations_actually_change_the_text(self):
        # A perturbation that is a no-op would silently inflate the robustness
        # score. identity is the deliberate exception.
        g = gold()
        for name in LABEL_INVARIANT:
            if name == "identity":
                continue
            fn, _ = PERTURBATIONS[name]
            text, _ = fn(SOURCE, g, random.Random(7))
            assert text != SOURCE, f"{name} was a no-op"

    def test_typos_never_corrupt_a_gold_token(self):
        g = gold()
        text, _ = p_typos(SOURCE, g, random.Random(3))
        protected = ["Acme", "Austin", "120,000", "150,000", "Python", "SQL"]
        for token in protected:
            assert token in text, f"typo perturbation corrupted protected token {token!r}"


class TestLabelTransforming:
    def test_strip_salary_nulls_the_salary_and_removes_the_evidence(self):
        text, new_gold = p_strip_salary(SOURCE, gold(), RNG)
        flat = flatten(new_gold)
        assert flat["salary.min_amount"] is None
        assert flat["salary.max_amount"] is None
        assert flat["salary.currency"] is None
        assert flat["salary.period"] is None
        assert "120,000" not in text
        # Everything else must survive -- otherwise the arm measures more than
        # salary abstention.
        assert flat["job_title"] == "Senior Backend Engineer"
        assert flat["years_experience_min"] == 5

    def test_strip_deadline_nulls_only_the_deadline(self):
        text, new_gold = p_strip_deadline(SOURCE, gold(), RNG)
        flat = flatten(new_gold)
        assert flat["application_deadline"] is None
        assert "deadline" not in text.lower()
        assert flat["salary.min_amount"] == 120000.0


class TestSuiteConstruction:
    def _examples(self, n: int = 4) -> list[dict]:
        return [{"posting_id": f"p{i}", "source_text": SOURCE, "target_json": to_target_json(gold())}
                for i in range(n)]

    def test_builds_every_arm_over_the_same_postings(self):
        suite = build_suite(self._examples(), list(PERTURBATIONS), n_per=3, seed=5)
        assert set(suite) == set(PERTURBATIONS)
        assert all(len(rows) == 3 for rows in suite.values())
        # Paired: the same underlying postings appear in every arm.
        bases = {name: sorted(r["posting_id"].split("::")[0] for r in rows)
                 for name, rows in suite.items()}
        assert len(set(map(tuple, bases.values()))) == 1

    def test_restricted_arm_is_declared(self):
        assert "truncate_tail" in RESTRICTED_FIELDS
        assert all(f in flatten(gold()) for f in RESTRICTED_FIELDS["truncate_tail"])
