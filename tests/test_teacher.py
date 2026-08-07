"""Teacher labeling protocol tests.

The self-consistency vote is the only quality mechanism standing between a
frontier model's guesses and a held-out test set that the whole project's
headline numbers are measured against. It is also the easiest thing in the repo
to break silently: a broken vote still produces labels, still produces agreement
scores, and still passes its own gate.
"""

from __future__ import annotations

import json

import pytest

from data.corpus import RawPosting
from data.generate_synthetic import (
    DEFAULT_BUDGETS,
    DEFAULT_MODELS,
    DEFAULT_OPENROUTER_MODEL,
    Budget,
    BudgetExhausted,
    ResponseCache,
    TeacherRefusal,
    _reject_if_unusable,
    label_posting,
    resolve_teacher,
    vote,
)
from data.schema import (
    JobPosting,
    Location,
    Salary,
    gemini_response_schema,
    openai_response_schema,
    to_target_json,
)

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


class TestPromptStability:
    """SCHEMA_CARD is hashed into the teacher cache key via build_user_prompt.

    Changing it mid-corpus is expensive and quiet: every cached response misses
    (measured: 3,000 gold responses, ~$0.77 of re-labeling), and if only part of
    the corpus is relabeled, train and test end up annotated under two different
    prompts -- the prompt drift schema.py's docstring exists to prevent.

    This pins the hash. If you intentionally change the prompt, update the
    expected value AND re-label the whole corpus, not just the part you noticed.
    """

    EXPECTED_SHA1 = "efd2d10956f11a9fe389a9937201b73e3788ae63"

    def test_prompt_template_hash_is_pinned(self):
        import hashlib

        from data.schema import SYSTEM_PROMPT, build_user_prompt

        digest = hashlib.sha1(
            (SYSTEM_PROMPT + build_user_prompt("PROBE")).encode()
        ).hexdigest()
        assert digest == self.EXPECTED_SHA1, (
            "the prompt changed -- every teacher cache entry is now a miss.\n"
            "If deliberate: update EXPECTED_SHA1 and re-label the FULL corpus,\n"
            "because a partial relabel splits train/test across two prompts."
        )


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


class TestOpenAISchemaTranslation:
    """The OpenRouter teacher decodes against a translated copy of the Gemini schema.

    Both teachers must be held to the same contract, so these assert the
    translation preserves it rather than just that it produces valid-looking JSON
    Schema. A drift here does not raise -- it produces labels with a quietly
    different shape.
    """

    def _schema(self) -> dict:
        return openai_response_schema()["json_schema"]["schema"]

    def test_strict_mode_is_requested(self):
        rf = openai_response_schema()
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["strict"] is True, \
            "without strict, response_format is a hint and JSON is no longer guaranteed"

    def test_same_top_level_fields_as_gemini(self):
        assert set(self._schema()["properties"]) == set(
            gemini_response_schema()["properties"]
        ), "the two teachers must decode against the same field set"

    def test_nullable_becomes_a_type_union(self):
        """`nullable: true` is an OpenAPI keyword; OpenAI-dialect validators ignore it."""
        props = self._schema()["properties"]
        assert props["company_name"]["type"] == ["string", "null"]
        assert props["years_experience_min"]["type"] == ["integer", "null"]
        assert props["job_title"]["type"] == "string", "non-nullable must stay scalar"

    def test_no_openapi_keywords_survive(self):
        """`nullable` and `propertyOrdering` are rejected or ignored downstream."""
        blob = json.dumps(self._schema())
        assert "nullable" not in blob
        assert "propertyOrdering" not in blob

    def test_every_object_is_closed_and_fully_required(self):
        """strict:true demands additionalProperties:false and all keys required."""
        def check(node: dict, path: str = "$") -> None:
            types = node["type"] if isinstance(node["type"], list) else [node["type"]]
            if "object" in types:
                assert node["additionalProperties"] is False, f"{path} is open"
                assert set(node["required"]) == set(node["properties"]), \
                    f"{path} must require every property"
                for k, v in node["properties"].items():
                    check(v, f"{path}.{k}")
            if "array" in types:
                check(node["items"], f"{path}[]")

        check(self._schema())

    def test_nullable_enum_admits_null(self):
        """A nullable enum that omits null from `enum` is unsatisfiable."""
        emp = self._schema()["properties"]["employment_type"]
        assert emp["type"] == ["string", "null"]
        assert None in emp["enum"]
        assert "full_time" in emp["enum"]

    def test_nested_objects_are_translated_too(self):
        loc = self._schema()["properties"]["location"]
        assert loc["additionalProperties"] is False
        assert loc["properties"]["city"]["type"] == ["string", "null"]


class TestBudgetDefaults:
    def test_every_teacher_choice_has_a_budget(self):
        """Missing entry = KeyError at runtime, after the corpus load."""
        assert set(DEFAULT_BUDGETS) == {"gemini", "openrouter", "mock"}

    def test_openrouter_default_is_the_conservative_case(self):
        """Free-tier cap is credit-gated; overshooting it costs a run of 429s."""
        _rpm, rpd = DEFAULT_BUDGETS["openrouter"]
        assert rpd < DEFAULT_BUDGETS["gemini"][1]

    def test_every_teacher_has_a_default_model(self):
        assert set(DEFAULT_MODELS) == set(DEFAULT_BUDGETS)


class TestTeacherResolution:
    """flag > .env > built-in. Every previous resolution bug in this module hid
    behind a corpus load, so this is tested directly."""

    @pytest.fixture(autouse=True)
    def _clear(self, monkeypatch):
        import config

        monkeypatch.setattr(config, "_loaded", True)  # skip reading a real .env
        for var in ("TEACHER", "TEACHER_MODEL"):
            monkeypatch.delenv(var, raising=False)

    def test_built_in_default_is_gemini(self):
        assert resolve_teacher() == ("gemini", DEFAULT_MODELS["gemini"])

    def test_env_selects_the_transport_and_its_default_model(self, monkeypatch):
        monkeypatch.setenv("TEACHER", "openrouter")
        assert resolve_teacher() == ("openrouter", DEFAULT_OPENROUTER_MODEL)

    def test_env_can_pin_the_model(self, monkeypatch):
        monkeypatch.setenv("TEACHER", "openrouter")
        monkeypatch.setenv("TEACHER_MODEL", "qwen/qwen3-30b-a3b-instruct-2507")
        assert resolve_teacher() == ("openrouter", "qwen/qwen3-30b-a3b-instruct-2507")

    def test_cli_beats_env(self, monkeypatch):
        monkeypatch.setenv("TEACHER", "openrouter")
        monkeypatch.setenv("TEACHER_MODEL", "from-env")
        assert resolve_teacher("mock", "from-cli") == ("mock", "from-cli")

    def test_unknown_transport_raises_rather_than_defaulting(self, monkeypatch):
        """A typo in .env must not silently route a PAID run through gemini."""
        monkeypatch.setenv("TEACHER", "openrotuer")
        with pytest.raises(ValueError, match="unknown teacher"):
            resolve_teacher()


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


class TestSetValuedVoting:
    """List fields are voted per ELEMENT, not as atomic values.

    Atomic voting made agreement a function of list length rather than of
    teacher reliability: on the first 500 real gold postings, `required_skills`
    averaged 0.59 agreement -- under the 0.67 gate -- on records that were
    accepted anyway, and list fields drove 87%/68%/43% of all low-agreement
    rejections.
    """

    def _with(self, skills: list[str]) -> JobPosting:
        return JobPosting(job_title="Engineer", required_skills=skills)

    def test_one_extra_item_does_not_destroy_agreement(self):
        """The exact shape that was tanking the yield."""
        samples = [self._with(["python", "sql"]),
                   self._with(["python", "sql"]),
                   self._with(["git", "python", "sql"])]
        voted, agreement = vote(samples)
        assert voted is not None
        assert voted.required_skills == ["python", "sql"], "minority item must not carry"
        assert agreement["required_skills"] > 0.67, "must clear the gate"

    def test_unanimous_lists_score_one(self):
        _v, agreement = vote([self._with(["python", "sql"])] * 3)
        assert agreement["required_skills"] == 1.0

    def test_unanimously_empty_is_agreement_not_absence(self):
        _v, agreement = vote([self._with([])] * 3)
        assert agreement["required_skills"] == 1.0

    def test_disjoint_lists_fail_the_gate(self):
        """Genuine disagreement must still be caught."""
        samples = [self._with(["python"]), self._with(["java"]), self._with(["go"])]
        voted, agreement = vote(samples)
        assert agreement["required_skills"] < 0.67
        assert voted is not None and voted.required_skills == [], \
            "nothing reaches a strict majority, so nothing survives"

    def test_item_in_exactly_half_is_dropped(self):
        """Contested items are dropped, not coin-flipped, for a held-out set."""
        samples = [self._with(["python", "sql"]), self._with(["python"])]
        voted, _a = vote(samples)
        assert voted is not None and voted.required_skills == ["python"]

    def test_scalar_fields_are_unaffected(self):
        """The dispatch must not change how non-list fields vote."""
        samples = [JobPosting(job_title="A", years_experience_min=5),
                   JobPosting(job_title="A", years_experience_min=5),
                   JobPosting(job_title="A", years_experience_min=3)]
        voted, agreement = vote(samples)
        assert voted is not None and voted.years_experience_min == 5
        assert agreement["years_experience_min"] == 2 / 3


class TestFailureIsolation:
    """One posting the teacher cannot handle must not end the run.

    Regression: the loop caught bare RuntimeError to stop on an exhausted daily
    budget. When the teacher began raising RuntimeError for per-posting problems
    too, the first unusable completion broke the loop and exited 0 -- a
    1,010-posting phase stopped after 26 and reported success.
    """

    def test_budget_exhaustion_is_its_own_type(self):
        assert issubclass(BudgetExhausted, RuntimeError)
        assert not issubclass(TeacherRefusal, BudgetExhausted)

    def test_budget_take_raises_the_stopping_type(self):
        b = Budget(requests_per_minute=100, requests_per_day=1)
        b.take()
        with pytest.raises(BudgetExhausted):
            b.take()

    def test_length_capped_completion_is_permanent_not_transient(self):
        """finish_reason='length' at temperature 0 repeats identically."""
        with pytest.raises(TeacherRefusal):
            _reject_if_unusable('{"job_title": "Eng"', structured=True, finish="length")

    def test_truncated_stream_is_transient_so_it_can_be_retried(self):
        """A cut stream is a network accident -- retryable, and NOT a refusal."""
        with pytest.raises(RuntimeError) as exc:
            _reject_if_unusable('{\n  "job_', structured=True, finish="stop")
        assert not isinstance(exc.value, TeacherRefusal)

    def test_valid_completion_passes(self):
        good = to_target_json(JobPosting(job_title="Engineer"))
        _reject_if_unusable(good, structured=True, finish="stop")

    def test_unstructured_mode_accepts_anything(self):
        """There the caller has explicitly opted into parse failures."""
        _reject_if_unusable("not json at all", structured=False, finish="stop")


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
