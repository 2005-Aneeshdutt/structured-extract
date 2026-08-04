"""Schema and parsing tests.

These are not decoration. Every number in the README flows through
`parse_prediction` and `flatten`; a silent bug in either would move every metric
at once and there would be nothing in the output to reveal it. The tests here
pin the behavior that the results depend on.
"""

from __future__ import annotations

import json

import pytest

from data.schema import (
    FIELD_SPECS,
    EmploymentType,
    JobPosting,
    Location,
    RemotePolicy,
    Salary,
    build_messages,
    build_user_prompt,
    canonicalize_terms,
    extract_json_block,
    flatten,
    gemini_response_schema,
    parse_prediction,
    to_target_json,
    truncate_source,
    ungrounded_fields,
)


def _minimal(**kw) -> JobPosting:
    return JobPosting(job_title="Engineer", **kw)


class TestCanonicalization:
    def test_lowercases_dedupes_and_sorts(self):
        assert canonicalize_terms(["Python", "python", "AWS", " SQL "]) == ["aws", "python", "sql"]

    def test_applies_aliases(self):
        assert canonicalize_terms(["JS", "K8s", "PostgreSQL"]) == ["javascript", "kubernetes", "postgresql"]

    def test_drops_degenerate_entries(self):
        out = canonicalize_terms(["x", "r", "c", "", "a" * 60])
        assert out == ["c", "r"]  # single chars other than r/c dropped, overlong dropped

    def test_respects_limit(self):
        assert len(canonicalize_terms([f"skill{i}" for i in range(30)], limit=15)) == 15


class TestValidation:
    def test_currency_is_normalized_not_rejected(self):
        assert Salary(currency="usd").currency == "USD"

    def test_bogus_currency_becomes_none(self):
        assert Salary(currency="dollars").currency is None

    def test_implausible_salary_rejected(self):
        with pytest.raises(Exception):
            Salary(min_amount=1_699_090_000_000)

    def test_bad_date_is_rejected_not_coerced(self):
        # A malformed date must surface as a schema failure, because that is a
        # real extraction error we want counted -- not quietly dropped to null.
        obj, err = parse_prediction(json.dumps({"job_title": "x", "application_deadline": "Dec 1 2025"}))
        assert obj is None and err.startswith("schema_error")

    def test_extra_keys_rejected(self):
        obj, err = parse_prediction(json.dumps({"job_title": "x", "salary_currency": "USD"}))
        assert obj is None and err.startswith("schema_error")


class TestLenientParsing:
    def test_strips_markdown_fences(self):
        raw = '```json\n{"job_title": "Cook"}\n```'
        obj, err = parse_prediction(raw)
        assert err is None and obj.job_title == "Cook"

    def test_ignores_leading_prose(self):
        raw = 'Here is the JSON you asked for:\n{"job_title": "Cook"}'
        obj, _ = parse_prediction(raw)
        assert obj is not None

    def test_braces_inside_strings_do_not_break_matching(self):
        raw = '{"job_title": "Dev {backend}", "company_name": null}'
        assert extract_json_block(raw) == raw

    def test_unbalanced_object_returns_none(self):
        assert extract_json_block('{"job_title": "x"') is None

    def test_no_json_at_all(self):
        obj, err = parse_prediction("I cannot help with that.")
        assert obj is None and err == "no_json"


class TestFlatten:
    def test_covers_every_scored_field(self):
        flat = flatten(_minimal())
        assert set(flat) == set(FIELD_SPECS)

    def test_reads_nested_leaves(self):
        obj = _minimal(location=Location(city="Austin", region="TX"),
                       salary=Salary(min_amount=1000, currency="USD"))
        flat = flatten(obj)
        assert flat["location.city"] == "Austin"
        assert flat["salary.min_amount"] == 1000.0
        assert flat["salary.period"] is None


class TestRoundTrip:
    def test_target_json_reparses_identically(self):
        obj = _minimal(required_skills=["Python", "aws"], salary=Salary(min_amount=100000, currency="usd"))
        s = to_target_json(obj)
        again, err = parse_prediction(s)
        assert err is None
        assert to_target_json(again) == s

    def test_field_order_is_stable(self):
        keys = list(json.loads(to_target_json(_minimal())))
        assert keys == list(JobPosting.model_fields)


class TestPromptContract:
    def test_truncation_is_applied_and_shared(self):
        long_text = "word " * 4000
        assert len(truncate_source(long_text)) <= 6010
        assert truncate_source(long_text) in build_user_prompt(long_text)

    def test_messages_shape(self):
        msgs = build_messages("posting", few_shot=[("src", '{"job_title":"x"}')])
        assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]

    def test_few_shot_reuses_the_same_template(self):
        msgs = build_messages("posting", few_shot=[("src", "{}")])
        assert msgs[1]["content"] == build_user_prompt("src")
        assert msgs[3]["content"] == build_user_prompt("posting")


class TestGeminiResponseSchema:
    """Structural checks on the constrained-decoding schema.

    Worth pinning because the failure mode is expensive and delayed: a malformed
    schema is rejected by the API on the first call, which is discovered only
    after starting a labeling run that is metered against a 1,500/day quota.
    Gemini's structured-output subset rejects `$ref`, `$defs` and `anyOf` -- all
    of which pydantic's `model_json_schema()` emits for optional and nested
    types, which is exactly why this schema is hand-built.
    """

    @staticmethod
    def _walk(node):
        yield node
        if isinstance(node, dict):
            for v in node.values():
                yield from TestGeminiResponseSchema._walk(v)
        elif isinstance(node, list):
            for v in node:
                yield from TestGeminiResponseSchema._walk(v)

    def test_no_unsupported_json_schema_keywords(self):
        banned = {"$ref", "$defs", "anyOf", "oneOf", "allOf", "definitions"}
        for node in self._walk(gemini_response_schema()):
            if isinstance(node, dict):
                assert not (banned & set(node)), f"unsupported keyword in {list(node)}"

    def test_covers_every_top_level_field(self):
        schema = gemini_response_schema()
        assert set(schema["properties"]) == set(JobPosting.model_fields)
        assert set(schema["required"]) == set(JobPosting.model_fields)

    def test_nested_objects_declare_their_leaves(self):
        props = gemini_response_schema()["properties"]
        assert set(props["location"]["properties"]) == {"city", "region", "country", "remote_policy"}
        assert set(props["salary"]["properties"]) == {"min_amount", "max_amount", "currency", "period"}

    def test_enums_match_the_pydantic_models(self):
        props = gemini_response_schema()["properties"]
        assert set(props["employment_type"]["enum"]) == {m.value for m in EmploymentType}
        assert set(props["location"]["properties"]["remote_policy"]["enum"]) == {
            m.value for m in RemotePolicy}

    def test_property_ordering_matches_canonical_field_order(self):
        # propertyOrdering pins generation order so self-consistency samples are
        # byte-comparable rather than differing only in key order.
        assert gemini_response_schema()["propertyOrdering"] == list(JobPosting.model_fields)

    def test_is_json_serializable(self):
        # It is sent over the wire; a stray enum or tuple would fail at call time.
        json.dumps(gemini_response_schema())


class TestGrounding:
    SOURCE = "Acme Corp is hiring in Austin. Pay is $120,000 per year. Requires 5 years of Python."

    def test_supported_values_pass(self):
        obj = _minimal(company_name="Acme Corp", location=Location(city="Austin"),
                       salary=Salary(min_amount=120000), years_experience_min=5,
                       required_skills=["python"])
        assert ungrounded_fields(obj, self.SOURCE) == []

    def test_region_is_canonicalized_on_construction(self):
        """Measured: 31.6% of gold labels wrote 'Texas', 68.4% wrote 'TX'.

        That split lived in the TRAINING TARGETS, so the student would learn to
        pick arbitrarily and then be marked wrong for picking the convention the
        gold label happened not to use. The validator runs on every JobPosting
        the codebase builds -- labels and predictions alike -- so normalizing
        cannot advantage one side of the comparison.
        """
        assert Location(region="Texas").region == "TX"
        assert Location(region="  texas ").region == "TX"
        assert Location(region="tx").region == "TX"
        assert Location(region="TX").region == "TX"
        assert Location(region="District of Columbia").region == "DC"

    def test_non_us_regions_pass_through(self):
        assert Location(region="Ontario").region == "Ontario"
        assert Location(region="Bavaria").region == "Bavaria"

    def test_blank_region_becomes_null(self):
        assert Location(region="   ").region is None
        assert Location(region=None).region is None

    def test_canonicalization_reaches_nested_parse(self):
        """It must apply through parse_prediction, not just direct construction."""
        obj, err = parse_prediction('{"job_title":"Eng","location":{"region":"California"}}')
        assert err is None and obj is not None
        assert obj.location.region == "CA"

    def test_canonicalized_region_is_grounded_by_either_spelling(self):
        """Canonicalization rewrites the value; grounding must not then punish it.

        "Texas" is normalized to "TX" on construction, so a literal substring
        check against a posting that only ever writes "Texas" would report a
        perfectly correct extraction as a hallucination. Cost of getting this
        wrong, measured: 20 extra rejected gold labels.
        """
        obj = _minimal(location=Location(region="Texas"))
        assert obj.location.region == "TX"
        assert "location.region" not in ungrounded_fields(obj, "Role based in Texas, hybrid.")
        assert "location.region" not in ungrounded_fields(obj, "Role based in Austin, TX.")

    def test_wrong_region_is_still_flagged(self):
        """The relaxation must not make the check toothless."""
        obj = _minimal(location=Location(region="California"))
        assert "location.region" in ungrounded_fields(obj, "Role based in Austin, TX.")

    def test_invented_company_is_flagged(self):
        assert "company_name" in ungrounded_fields(_minimal(company_name="Globex"), self.SOURCE)

    def test_invented_salary_is_flagged(self):
        assert "salary.min_amount" in ungrounded_fields(
            _minimal(salary=Salary(min_amount=95000)), self.SOURCE)

    def test_k_suffix_counts_as_support(self):
        obj = _minimal(salary=Salary(min_amount=120000))
        assert "salary.min_amount" not in ungrounded_fields(obj, "Salary: $120k per year")

    def test_checks_against_truncated_text_only(self):
        # A value that appears only past the truncation budget was never visible
        # to the model, so asserting it IS a hallucination and must be flagged.
        padded = "filler " * 2000 + " Globex Corporation"
        assert "company_name" in ungrounded_fields(_minimal(company_name="Globex Corporation"), padded)
