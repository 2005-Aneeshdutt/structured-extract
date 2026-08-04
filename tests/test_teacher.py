"""Teacher labeling protocol tests.

The self-consistency vote is the only quality mechanism standing between a
frontier model's guesses and a held-out test set that the whole project's
headline numbers are measured against. It is also the easiest thing in the repo
to break silently: a broken vote still produces labels, still produces agreement
scores, and still passes its own gate.
"""

from __future__ import annotations

import json

from data.corpus import RawPosting
from data.generate_synthetic import ResponseCache, label_posting, vote
from data.schema import JobPosting, Location, Salary, to_target_json

SOURCE = ("Acme is hiring a Backend Engineer in Austin, TX. Full-time, senior. "
          "$120,000 per year. Requires 5 years of Python and SQL. Bachelor's degree.")


class TestCacheKey:
    """The bug this file exists for.

    Self-consistency draws k samples at the same temperature. Without the sample
    index in the cache key they collide, and samples 2..k are served as
    byte-identical copies of sample 1 -- so the vote sees k-1 duplicates, the
    duplicated value always wins, and genuine teacher disagreement becomes
    unobservable while agreement scores still look reasonable.
    """

    def test_same_temperature_different_sample_yields_different_key(self):
        a = ResponseCache.key("gemini-2.0-flash", 0.4, "prompt", sample_idx=1)
        b = ResponseCache.key("gemini-2.0-flash", 0.4, "prompt", sample_idx=2)
        assert a != b, "samples at the same temperature must not share a cache entry"

    def test_identical_calls_still_hit_cache(self):
        a = ResponseCache.key("gemini-2.0-flash", 0.4, "prompt", sample_idx=1)
        b = ResponseCache.key("gemini-2.0-flash", 0.4, "prompt", sample_idx=1)
        assert a == b, "resumability depends on identical calls sharing a key"

    def test_key_varies_with_every_component(self):
        base = ResponseCache.key("m", 0.0, "p", 0)
        assert ResponseCache.key("other", 0.0, "p", 0) != base
        assert ResponseCache.key("m", 0.4, "p", 0) != base
        assert ResponseCache.key("m", 0.0, "other", 0) != base
        assert ResponseCache.key("m", 0.0, "p", 1) != base


class TestSampleIndependence:
    def test_three_samples_produce_three_distinct_calls(self):
        """A counting teacher proves each sample reaches the model separately."""
        seen: list[tuple[float, int]] = []

        def counting_teacher(text: str, temperature: float = 0.0, sample_idx: int = 0) -> str:
            seen.append((temperature, sample_idx))
            # Vary one field per sample so a collapsed vote would be detectable.
            return to_target_json(JobPosting(job_title=f"Engineer {sample_idx}"))

        posting = RawPosting("p1", SOURCE, {})
        label_posting(posting, counting_teacher, n_samples=3, temperature=0.4)

        assert len(seen) == 3
        assert len({s for _t, s in seen}) == 3, "each sample needs its own index"
        assert seen[0][0] == 0.0, "sample 0 must be greedy"
        assert all(t == 0.4 for t, _s in seen[1:]), "samples 1..k share the temperature"


class TestVoting:
    def _obj(self, **kw) -> JobPosting:
        base = {
            "job_title": "Backend Engineer",
            "location": Location(city="Austin", region="TX"),
            "salary": Salary(min_amount=120000, currency="USD", period="yearly"),
            "years_experience_min": 5,
        }
        return JobPosting(**{**base, **kw})

    def test_majority_wins_per_field(self):
        a, b, c = self._obj(), self._obj(), self._obj(years_experience_min=3)
        voted, agreement = vote([a, b, c])
        assert voted is not None
        assert voted.years_experience_min == 5
        assert agreement["years_experience_min"] == 1 / 3 or agreement["years_experience_min"] == 2 / 3

    def test_unanimous_fields_report_full_agreement(self):
        voted, agreement = vote([self._obj(), self._obj(), self._obj()])
        assert voted is not None
        assert agreement["job_title"] == 1.0
        assert agreement["salary.min_amount"] == 1.0

    def test_disagreement_is_visible_in_the_scores(self):
        """If every sample differs, agreement must reflect that rather than 1.0."""
        samples = [self._obj(years_experience_min=n) for n in (3, 5, 7)]
        _voted, agreement = vote(samples)
        assert agreement["years_experience_min"] < 0.67, \
            "three-way disagreement must fall below the gate"

    def test_empty_input_is_handled(self):
        voted, agreement = vote([])
        assert voted is None and agreement == {}


class TestLabelFunnel:
    def test_low_agreement_examples_are_rejected_not_repaired(self):
        """A contested example must be dropped, never averaged into agreement."""
        values = iter([3, 5, 7])

        def disagreeing_teacher(text: str, temperature: float = 0.0, sample_idx: int = 0) -> str:
            return to_target_json(JobPosting(job_title="Engineer",
                                             years_experience_min=next(values)))

        rec = label_posting(RawPosting("p1", SOURCE, {}), disagreeing_teacher,
                            n_samples=3, temperature=0.4)
        assert rec["status"] == "low_agreement"
        assert "target_json" not in rec, "rejected examples must not carry a label"

    def test_agreeing_examples_pass(self):
        def agreeing_teacher(text: str, temperature: float = 0.0, sample_idx: int = 0) -> str:
            return to_target_json(JobPosting(job_title="Backend Engineer",
                                             years_experience_min=5,
                                             required_skills=["python", "sql"]))

        rec = label_posting(RawPosting("p1", SOURCE, {}), agreeing_teacher,
                            n_samples=3, temperature=0.4)
        assert rec["status"] == "ok"
        assert json.loads(rec["target_json"])["years_experience_min"] == 5
