"""Canonical extraction schema -- the single source of truth for the whole project.

This module is imported by data generation, training, evaluation, quantization
verification and the Gradio app. It is also *vendored verbatim* into the Kaggle
training notebook.

WHY that matters: the #1 way a fine-tuning project produces numbers that do not
replicate is prompt drift -- the instruction template used at train time differs
by a stray newline from the one used at eval time, and the reported lift is
partly an artifact of that mismatch. Keeping the system prompt, the user
template and the JSON serialization in one file that every stage imports makes
that class of bug impossible rather than merely unlikely.

Hard dependency budget: pydantic only. No torch, no datasets, no transformers --
so it can be pasted into a Kaggle cell without dragging in the world.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

SCHEMA_VERSION: Final = "1.0.0"

#: Maximum items per list field. ONE definition, consumed by three places that
#: previously each carried their own copy of the number (or, worse, only a
#: sentence about it): the pydantic validator that truncates, the JSON schema
#: `maxItems` that stops the model generating past it, and the quality filter
#: that rejects overflow. When those drifted, the grammar allowed what the
#: validator would not produce and the filter silently deleted the difference.
LIST_CAPS: Final[dict[str, int]] = {
    "required_skills": 15,
    "preferred_skills": 10,
    "benefits": 10,
}

# ---------------------------------------------------------------------------
# Controlled vocabularies
#
# WHY closed enums instead of free strings for these five fields: they turn a
# generation problem into a classification problem at eval time. A free-text
# `seniority_level` would force fuzzy matching ("Sr." vs "Senior" vs "senior")
# and every reported accuracy number would then depend on the fuzz threshold --
# an immediate credibility problem in an interview. Closed vocabularies make
# per-field accuracy exact and arguable-with.
# ---------------------------------------------------------------------------


class RemotePolicy(str, Enum):
    ONSITE = "onsite"
    HYBRID = "hybrid"
    REMOTE = "remote"


class EmploymentType(str, Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERNSHIP = "internship"
    TEMPORARY = "temporary"
    VOLUNTEER = "volunteer"


class SeniorityLevel(str, Enum):
    INTERN = "intern"
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"
    EXECUTIVE = "executive"


class EducationLevel(str, Enum):
    NONE = "none"
    HIGH_SCHOOL = "high_school"
    ASSOCIATE = "associate"
    BACHELOR = "bachelor"
    MASTER = "master"
    DOCTORATE = "doctorate"


class PayPeriod(str, Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


# ---------------------------------------------------------------------------
# Nested objects
# ---------------------------------------------------------------------------


#: US state / territory name -> USPS code. Exists because the corpus is
#: US-dominated and the teacher used both conventions interchangeably: measured
#: over 297 gold labels carrying a region, 31.6% wrote "Texas" and 68.4% wrote
#: "TX". That is not a cosmetic split. It is inconsistency *in the training
#: targets*, so the student learns to pick arbitrarily, and then the evaluator
#: marks it wrong whenever it picks the convention the gold label did not use --
#: a self-inflicted accuracy loss on a tier-2 field.
#:
#: The original field description ("full name or code as written") is what
#: invited it, so the description now pins the convention and this map enforces
#: it. Non-US regions pass through untouched.
#:
#: Enforced by a VALIDATOR rather than by an instruction in SCHEMA_CARD, and
#: that is deliberate on two counts. A validator is deterministic where a prompt
#: instruction is merely a request, and it applies identically to labels and to
#: predictions, so normalizing cannot advantage one side of the comparison.
#: SCHEMA_CARD is also load-bearing in a way that is easy to miss: it is part of
#: `build_user_prompt`, which is hashed into the teacher cache key AND is the
#: prompt used at train and eval time. Editing it invalidates every cached
#: response (measured: 3,000 gold responses, ~$0.77 of re-labeling) and, worse,
#: silently splits the corpus across two prompts if half of it is already
#: labeled -- the exact drift this module's docstring opens by warning about.
US_STATE_CODES: Final[dict[str, str]] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC", "washington dc": "DC", "washington d.c.": "DC",
    "puerto rico": "PR",
}

#: Reverse lookup, for comparing our canonical code against a source string that
#: spelled the state out (LinkedIn writes both "Austin, TX" and "Austin, Texas
#: Metropolitan Area"). Built from the forward map so the two cannot drift.
US_STATE_NAMES: Final[dict[str, str]] = {v: k for k, v in US_STATE_CODES.items()}


def canonicalize_region(value: str | None) -> str | None:
    """Normalize a US state to its USPS code; pass anything else through.

    Applied as a pydantic validator, so it runs on EVERY JobPosting the codebase
    constructs -- teacher labels, training targets, and student predictions at
    eval time alike. That uniformity is the point: normalizing only the labels
    would penalize a model whose output was right but spelled differently.
    """
    if value is None:
        return None
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        return None
    code = US_STATE_CODES.get(cleaned.lower())
    if code:
        return code
    # Already a bare 2-letter code for a real state ("tx" -> "TX"). Guarded by
    # the membership test so a genuine 2-letter non-US region is left alone.
    if len(cleaned) == 2 and cleaned.upper() in US_STATE_NAMES:
        return cleaned.upper()
    return cleaned


class Location(BaseModel):
    """Nested object #1. Deliberately includes an enum sub-field.

    WHY nested at all: flat schemas are the easy case. A senior reviewer will ask
    whether a 1.5B model can hold a two-level structure without dropping a brace,
    and "we measured it" is a better answer than "we avoided it".
    """

    model_config = ConfigDict(extra="forbid")

    city: str | None = Field(None, description="City name only, e.g. 'Austin'. Null if not stated.")
    region: str | None = Field(
        None,
        description="State / province / region. US states as the 2-letter code, e.g. 'TX'.",
    )

    country: str | None = Field(None, description="ISO 3166 English country name, e.g. 'United States'.")
    remote_policy: RemotePolicy | None = Field(
        None, description="onsite | hybrid | remote. Null if the posting never says."
    )

    @field_validator("region")
    @classmethod
    def _canonical_region(cls, v: str | None) -> str | None:
        return canonicalize_region(v)


class Salary(BaseModel):
    """Nested object #2, and the hardest field in the schema.

    Requires three distinct capabilities in one shot:
      1. extraction   -- find "$120,000 - $150,000 per year" in 4 KB of prose
      2. normalization-- "$120k" -> 120000.0, "£45,000 p.a." -> GBP/yearly
      3. restraint    -- most postings have NO salary; emitting one is a
                         hallucination, and this field is where base models
                         hallucinate most.
    """

    model_config = ConfigDict(extra="forbid")

    min_amount: float | None = Field(None, description="Lower bound as a plain number, no symbols or commas.")
    max_amount: float | None = Field(None, description="Upper bound as a plain number. Equal to min if a single figure is given.")
    currency: str | None = Field(None, description="ISO-4217 code, e.g. 'USD', 'GBP', 'EUR'.")
    period: PayPeriod | None = Field(None, description="hourly | daily | weekly | monthly | yearly.")

    @field_validator("currency")
    @classmethod
    def _upper_iso(cls, v: str | None) -> str | None:
        # Teacher and student both occasionally emit "usd" / "Usd". Case is not
        # a semantic error, so we normalize rather than reject -- otherwise the
        # schema-compliance metric would measure capitalization, not structure.
        if v is None:
            return None
        v = v.strip().upper()
        return v if re.fullmatch(r"[A-Z]{3}", v) else None

    @field_validator("min_amount", "max_amount")
    @classmethod
    def _sane_magnitude(cls, v: float | None) -> float | None:
        # Guards against the classic "120" (meaning 120k) and against absurd
        # parses like 1699090000000 (a scraped epoch leaking into the field).
        if v is None:
            return None
        if not (0 < v < 1e9):
            raise ValueError(f"implausible salary magnitude: {v}")
        return round(float(v), 2)


# ---------------------------------------------------------------------------
# Root object
# ---------------------------------------------------------------------------


class JobPosting(BaseModel):
    """12 fields spanning 4 difficulty tiers -- see FIELD_SPECS for the rationale.

    Field order is load-bearing: pydantic preserves declaration order on dump, so
    every training target and every prediction serializes identically. Easy,
    high-signal fields come first so that if generation truncates, the cheap
    fields survive; this also gives the model a stable "curriculum" within each
    sequence.
    """

    model_config = ConfigDict(extra="forbid")

    # -- tier 1: near-verbatim extraction ---------------------------------
    job_title: str = Field(..., description="The role title as advertised, cleaned of seniority noise markers like '(Remote)' or req IDs.")
    company_name: str | None = Field(None, description="Hiring company. Null if the text never names it.")

    # -- tier 2: nested structure -----------------------------------------
    location: Location = Field(default_factory=Location)

    # -- tier 3: classification into a closed vocabulary --------------------
    employment_type: EmploymentType | None = None
    seniority_level: SeniorityLevel | None = Field(
        None, description="Infer from title markers ('Staff', 'II', 'Jr') and stated experience. Null if genuinely unclear."
    )
    education_requirement: EducationLevel | None = Field(
        None, description="Highest level REQUIRED (not preferred). 'none' means explicitly no degree needed; null means unstated."
    )

    # -- tier 4: extraction + normalization + restraint ---------------------
    salary: Salary = Field(default_factory=Salary)
    years_experience_min: int | None = Field(
        None, ge=0, le=50, description="Minimum years required as an integer. '3-5 years' -> 3. Null unless a number is stated."
    )
    application_deadline: str | None = Field(
        None, description="ISO-8601 date YYYY-MM-DD. Null unless an explicit deadline date appears."
    )

    # -- tier 3b: set extraction -------------------------------------------
    required_skills: list[str] = Field(default_factory=list, description="Hard skills/tools explicitly required. Lowercase. Max 15.")
    preferred_skills: list[str] = Field(default_factory=list, description="Skills listed as nice-to-have/preferred/bonus ONLY. Max 10.")
    benefits: list[str] = Field(default_factory=list, description="Perks/benefits explicitly offered. Lowercase short phrases. Max 10.")

    # ---- validators ------------------------------------------------------

    @field_validator("job_title")
    @classmethod
    def _nonempty_title(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("job_title must be non-empty")
        return v[:120]

    @field_validator("company_name")
    @classmethod
    def _clean_company(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = " ".join(v.split())
        return v[:120] or None

    @field_validator("application_deadline")
    @classmethod
    def _iso_date(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        # Reject rather than coerce: a malformed date is a real extraction
        # failure and should show up in the schema-compliance metric.
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            raise ValueError(f"application_deadline must be YYYY-MM-DD, got {v!r}")
        return v

    @field_validator("required_skills", "preferred_skills", "benefits")
    @classmethod
    def _canonical_list(cls, v: list[str], info: ValidationInfo) -> list[str]:
        # The `limit` was documented in three places and passed in none: the
        # field descriptions say "Max 15", canonicalize_terms accepts a limit,
        # and prepare_dataset rejects overflow with the comment "should be
        # impossible post-validation". It fired 486 times -- 11.5% of the corpus
        # discarded whole, every other field on those postings lost with it,
        # because the one place that could enforce the cap did not.
        return canonicalize_terms(v, limit=LIST_CAPS[info.field_name])


# ---------------------------------------------------------------------------
# Term canonicalization
#
# WHY: "PyTorch", "pytorch", "Py-Torch" and " pytorch " are the same skill. If we
# leave that to the metric we end up hand-tuning a fuzzy threshold and the
# headline F1 becomes a function of that threshold. Instead we normalize once, in
# the schema, so *training targets* are already canonical and the metric can stay
# exact-match on sets. The alias table is deliberately small and only covers
# unambiguous, high-frequency variants -- it is not a knowledge base.
# ---------------------------------------------------------------------------

SKILL_ALIASES: Final[dict[str, str]] = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "k8s": "kubernetes",
    "gcp": "google cloud",
    "postgres": "postgresql",
    "ms excel": "excel",
    "microsoft excel": "excel",
    "ms office": "microsoft office",
    "node": "node.js",
    "nodejs": "node.js",
    "reactjs": "react",
    "react.js": "react",
    "c sharp": "c#",
    "golang": "go",
    "ci cd": "ci/cd",
    "cicd": "ci/cd",
    "restful apis": "rest apis",
    "rest api": "rest apis",
    "ml": "machine learning",
    "nlp": "natural language processing",
}

MAX_LIST_LEN: Final[dict[str, int]] = {
    "required_skills": 15,
    "preferred_skills": 10,
    "benefits": 10,
}


def canonicalize_terms(terms: list[str], *, limit: int | None = None) -> list[str]:
    """Lowercase, de-punctuate, alias-map, dedupe, and sort a list of terms.

    Sorting is intentional. Set membership is what we actually care about, and
    the metric scores these fields as sets -- but the *training target* has to
    pick some order. An arbitrary order would force the model to spend capacity
    memorizing the teacher's ordering quirks. Alphabetical order is a free,
    deterministic convention that removes that noise.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in terms:
        if not isinstance(raw, str):
            continue
        t = " ".join(raw.lower().replace("_", " ").split())
        t = t.strip(" .,;:-•*/()[]")
        t = SKILL_ALIASES.get(t, t)
        # Drop degenerate entries: single characters (except real ones like "r",
        # "c") and sentence-length blobs the teacher sometimes emits.
        if not t or len(t) > 40:
            continue
        if len(t) == 1 and t not in {"r", "c"}:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    out.sort()
    return out[:limit] if limit else out


# ---------------------------------------------------------------------------
# Field specification table -- drives the evaluation harness.
#
# Keeping tier/comparator metadata *here* rather than in eval/ means the metric
# definition and the schema definition cannot drift apart. `comparator` selects
# the scoring function; `tier` is only for reporting (it lets the results table
# show that the model is strong on tier-1 and weak on tier-4, which is the
# interesting story).
# ---------------------------------------------------------------------------

Comparator = Literal["string", "categorical", "numeric", "set", "date"]

FIELD_SPECS: Final[dict[str, dict[str, Any]]] = {
    "job_title":              {"tier": 1, "comparator": "string",      "nullable": False},
    "company_name":           {"tier": 1, "comparator": "string",      "nullable": True},
    "location.city":          {"tier": 2, "comparator": "string",      "nullable": True},
    "location.region":        {"tier": 2, "comparator": "string",      "nullable": True},
    "location.country":       {"tier": 2, "comparator": "string",      "nullable": True},
    "location.remote_policy": {"tier": 2, "comparator": "categorical", "nullable": True},
    "employment_type":        {"tier": 3, "comparator": "categorical", "nullable": True},
    "seniority_level":        {"tier": 3, "comparator": "categorical", "nullable": True},
    "education_requirement":  {"tier": 3, "comparator": "categorical", "nullable": True},
    "required_skills":        {"tier": 3, "comparator": "set",         "nullable": False},
    "preferred_skills":       {"tier": 3, "comparator": "set",         "nullable": False},
    "benefits":               {"tier": 3, "comparator": "set",         "nullable": False},
    "salary.min_amount":      {"tier": 4, "comparator": "numeric",     "nullable": True},
    "salary.max_amount":      {"tier": 4, "comparator": "numeric",     "nullable": True},
    "salary.currency":        {"tier": 4, "comparator": "categorical", "nullable": True},
    "salary.period":          {"tier": 4, "comparator": "categorical", "nullable": True},
    "years_experience_min":   {"tier": 4, "comparator": "numeric",     "nullable": True},
    "application_deadline":   {"tier": 4, "comparator": "date",        "nullable": True},
}

TIER_NAMES: Final[dict[int, str]] = {
    1: "verbatim extraction",
    2: "nested structure",
    3: "closed-vocabulary / set",
    4: "normalization + restraint",
}

#: Fields for which the source corpus carries *independent, non-LLM* ground truth
#: (LinkedIn's own structured form fields). Used by the label-quality audit in
#: generate_synthetic.py. See README "Are your labels any good?".
PLATFORM_AUDITABLE_FIELDS: Final[tuple[str, ...]] = (
    "job_title",
    "location.city",
    "location.region",
    "location.remote_policy",
    "employment_type",
    "seniority_level",
    "salary.min_amount",
    "salary.max_amount",
    "salary.currency",
    "salary.period",
)


# ---------------------------------------------------------------------------
# The prompt. Frozen. Identical for teacher, base zero-shot, base few-shot,
# fine-tuned student and the Gradio app.
#
# WHY hand-written and terse rather than a long rulebook: the base-model
# comparison has to be fair. A 900-token prompt would inflate base-model scores
# (prompt engineering, not fine-tuning) and simultaneously waste T4 sequence
# budget on every training example. This prompt states the contract and the six
# rules that a base model demonstrably cannot follow without training -- most
# importantly the null-over-guess rule, which is where the lift comes from.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: Final = (
    "You are a precise information extraction engine. You read a raw job posting "
    "and return exactly one JSON object matching the given schema.\n"
    "Rules:\n"
    "1. Output JSON only. No markdown fences, no commentary, no trailing text.\n"
    "2. Use null for any field the posting does not state. Never guess.\n"
    "3. Never invent values that are not supported by the text.\n"
    "4. Normalize numbers to plain digits (120000, not \"$120k\").\n"
    "5. Lists are lowercase, deduplicated, and alphabetically sorted.\n"
    "6. Emit every schema key, even when the value is null or []."
)

#: Compact schema card shown to the model. Hand-written rather than
#: `model_json_schema()` because pydantic emits $defs/anyOf, which costs ~3x the
#: tokens and which small models handle noticeably worse than a plain sketch.
SCHEMA_CARD: Final = """{
  "job_title": string,
  "company_name": string|null,
  "location": {"city": string|null, "region": string|null, "country": string|null,
               "remote_policy": "onsite"|"hybrid"|"remote"|null},
  "employment_type": "full_time"|"part_time"|"contract"|"internship"|"temporary"|"volunteer"|null,
  "seniority_level": "intern"|"entry"|"mid"|"senior"|"lead"|"principal"|"executive"|null,
  "education_requirement": "none"|"high_school"|"associate"|"bachelor"|"master"|"doctorate"|null,
  "salary": {"min_amount": number|null, "max_amount": number|null,
             "currency": string|null,
             "period": "hourly"|"daily"|"weekly"|"monthly"|"yearly"|null},
  "years_experience_min": integer|null,
  "application_deadline": "YYYY-MM-DD"|null,
  "required_skills": [string],
  "preferred_skills": [string],
  "benefits": [string]
}"""

MAX_SOURCE_CHARS: Final = 6000
"""Truncation budget for the posting text.

Set from the measured corpus distribution, not guessed. Over all 25,186 cleaned,
deduplicated postings: p50 = 3,481 chars, p90 = 6,521, p95 = 7,567, max = 19,903,
mean = 3,800. A 6,000-char cap leaves **86.1%** of postings untouched; 8,000
would leave 96.3%. (Reproduce with `corpus.describe_lengths`.)

We take 6,000 anyway, because that is what pins the full training sequence
(schema card + posting + JSON target) under 2,048 tokens, and 2,048 is the
sequence length at which a 3-epoch LoRA run on 4k examples fits comfortably
inside one Kaggle T4 session. Attention is quadratic in sequence length, so the
8,000-char variant costs roughly 1.6x the wall clock to recover 12% more of the
tail -- and the tail of a job posting is EEO boilerplate, not label-bearing text.

Truncation is head-first for the same reason: title, company, location and
salary cluster in the first screen. Both teacher and student see the *same*
truncated text (both go through `build_user_prompt`), so there is no
train/label mismatch -- the labels describe exactly what the model can see.

`training/configs/` exposes `max_seq_length` if you want to revisit the trade."""


def truncate_source(posting_text: str) -> str:
    """Apply the shared truncation budget.

    Exposed separately so the *grounding check* in data generation and the
    *hallucination metric* in eval can verify against the text the model actually
    saw, rather than the full posting. Checking against untruncated text would
    make both measurements silently lenient: a value invented by the model could
    be excused because it happens to appear in a tail the model never received.
    """
    text = posting_text.strip()
    if len(text) <= MAX_SOURCE_CHARS:
        return text
    return text[:MAX_SOURCE_CHARS].rsplit(" ", 1)[0] + " ..."


def build_user_prompt(posting_text: str) -> str:
    """Render the user turn. The ONLY place this template exists."""
    return f"Schema:\n{SCHEMA_CARD}\n\nJob posting:\n<<<\n{truncate_source(posting_text)}\n>>>\n\nJSON:"


def build_messages(posting_text: str, few_shot: list[tuple[str, str]] | None = None) -> list[dict[str, str]]:
    """Chat-format messages. `few_shot` is a list of (posting_text, json_str).

    The few-shot variant reuses the exact same system prompt and user template,
    so the 0-shot vs 3-shot comparison isolates the effect of the exemplars and
    nothing else.
    """
    msgs: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for src, tgt in few_shot or []:
        msgs.append({"role": "user", "content": build_user_prompt(src)})
        msgs.append({"role": "assistant", "content": tgt})
    msgs.append({"role": "user", "content": build_user_prompt(posting_text)})
    return msgs


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

JSON_DUMP_KWARGS: Final[dict[str, Any]] = {"ensure_ascii": False, "separators": (",", ":")}
"""Compact, no whitespace.

WHY: the target is ~40% shorter than pretty-printed JSON. On 4k examples x 3
epochs that is a material chunk of T4 time spent generating indentation, and
every whitespace token is another token the model can get wrong. The Gradio app
pretty-prints for display, which is a presentation concern, not a data one."""


def to_target_json(obj: JobPosting) -> str:
    """Serialize a validated object to the exact string used as a training target."""
    return json.dumps(obj.model_dump(mode="json"), **JSON_DUMP_KWARGS)


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def extract_json_block(text: str) -> str | None:
    """Pull the first balanced top-level JSON object out of raw model output.

    Base models wrap output in ``` fences or prepend "Here is the JSON:". Our
    schema-compliance metric would otherwise be measuring markdown habits rather
    than structural ability. Stripping fences and brace-matching is the standard
    lenient-parse step; we report BOTH strict (raw parse) and lenient rates in
    the eval so the reader can see how much of the base model's failure is
    cosmetic. Returns None if no balanced object exists.
    """
    text = _FENCE_RE.sub("", text)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_prediction(raw: str) -> tuple[JobPosting | None, str | None]:
    """Lenient parse of a model completion into a validated JobPosting.

    Returns (obj, None) on success or (None, reason) on failure. `reason` is one
    of: 'no_json', 'json_decode_error', 'schema_error:<detail>' -- these become
    the failure taxonomy in the failure analysis, so they are deliberately coarse
    and mutually exclusive.
    """
    block = extract_json_block(raw)
    if block is None:
        return None, "no_json"
    try:
        payload = json.loads(block)
    except json.JSONDecodeError as e:
        return None, f"json_decode_error:{e.msg}"
    if not isinstance(payload, dict):
        return None, "json_decode_error:not_an_object"
    try:
        return JobPosting.model_validate(payload), None
    except Exception as e:  # pydantic ValidationError, but keep it broad
        first = str(e).splitlines()
        detail = first[1].strip() if len(first) > 1 else first[0][:80]
        return None, f"schema_error:{detail[:80]}"


# ---------------------------------------------------------------------------
# Gemini structured-output schema
# ---------------------------------------------------------------------------


def gemini_response_schema() -> dict[str, Any]:
    """OpenAPI-subset schema for Gemini's `response_schema` constrained decoding.

    Hand-built rather than derived from `model_json_schema()` because Gemini
    rejects `$ref`/`$defs`/`anyOf`, which pydantic always emits for optional and
    nested types. Nullability is expressed with `nullable: true` instead.

    Using constrained decoding for the *teacher* is the whole reason the label
    set is clean: it makes 'invalid JSON from the teacher' structurally
    impossible, so every rejected label is a semantic rejection we chose, not a
    parsing accident.
    """
    def s(t: str, **kw: Any) -> dict[str, Any]:
        return {"type": t, **kw}

    # Every bound below mirrors a constraint the pydantic model already
    # enforces. Where the two disagree, the grammar wins at generation time and
    # pydantic wins at validation time -- so the model is free to emit something
    # that is then thrown away, and the call is billed either way. Measured on a
    # live run before these were added: 10.6% of postings failed, split between
    # repetition loops with no legal stopping token and integers outside the
    # validator's range.
    #
    # `maxLength` on strings is the counterpart to `maxItems` on arrays: without
    # it a single string value is also an unbounded region the model can loop
    # inside. Bounds are generous -- they exist to make runaway generation
    # impossible, not to trim honest values.
    TITLE, NAME, PLACE, TERM = 150, 150, 100, 40

    enum_of = lambda e: s("string", enum=[m.value for m in e], nullable=True)  # noqa: E731

    return s(
        "object",
        properties={
            "job_title": s("string", maxLength=TITLE),
            "company_name": s("string", nullable=True, maxLength=NAME),
            "location": s(
                "object",
                properties={
                    "city": s("string", nullable=True, maxLength=PLACE),
                    "region": s("string", nullable=True, maxLength=PLACE),
                    "country": s("string", nullable=True, maxLength=PLACE),
                    "remote_policy": enum_of(RemotePolicy),
                },
                required=["city", "region", "country", "remote_policy"],
            ),
            "employment_type": enum_of(EmploymentType),
            "seniority_level": enum_of(SeniorityLevel),
            "education_requirement": enum_of(EducationLevel),
            "salary": s(
                "object",
                properties={
                    # Mirrors _sane_magnitude's 0 < v < 1e9, which rejects both
                    # "120" meaning 120k and a scraped epoch leaking into the field.
                    "min_amount": s("number", nullable=True, minimum=1, maximum=999_999_999),
                    "max_amount": s("number", nullable=True, minimum=1, maximum=999_999_999),
                    "currency": s("string", nullable=True, maxLength=3),
                    "period": enum_of(PayPeriod),
                },
                required=["min_amount", "max_amount", "currency", "period"],
            ),
            # Mirrors the field's ge=0, le=50. Without these the grammar admits
            # any integer and pydantic then rejects it, which bills a call to
            # produce a label that is discarded.
            "years_experience_min": s("integer", nullable=True, minimum=0, maximum=50),
            "application_deadline": s("string", nullable=True, maxLength=10),
            # maxItems is load-bearing, not documentation. Without it the
            # grammar permits an unbounded array, so a model that slips into a
            # repetition loop has NO legal stopping token until it chooses to
            # close the array -- and it does not. Observed on a real posting:
            # "schedules" repeated for 8,192 tokens, 23KB of output, 13x the
            # cost of a normal call, and unparseable at the end of it.
            # Constrained decoding guarantees shape, and an unbounded array is a
            # shape that admits an infinite document.
            #
            # Caps match the field descriptions, which previously said "Max 15"
            # only in prose the model never saw -- descriptions are not part of
            # this hand-built schema. Verified enforced by the provider:
            # finish_reason went length -> stop, 8192 -> 388 completion tokens.
            # Item maxLength matches canonicalize_terms, which already drops any
            # term over 40 chars -- so the grammar now refuses to generate what
            # the validator would silently discard.
            "required_skills": s("array", items=s("string", maxLength=TERM),
                                 maxItems=LIST_CAPS["required_skills"]),
            "preferred_skills": s("array", items=s("string", maxLength=TERM),
                                  maxItems=LIST_CAPS["preferred_skills"]),
            "benefits": s("array", items=s("string", maxLength=TERM),
                          maxItems=LIST_CAPS["benefits"]),
        },
        required=list(JobPosting.model_fields.keys()),
        # propertyOrdering pins generation order to our canonical field order so
        # teacher output is byte-comparable across self-consistency samples.
        propertyOrdering=list(JobPosting.model_fields.keys()),
    )


def _strict_json_schema(node: dict[str, Any]) -> dict[str, Any]:
    """Rewrite one Gemini/OpenAPI node as strict JSON Schema (OpenAI dialect).

    Derived from `gemini_response_schema()` rather than written out a second time
    so the two teachers are provably decoding against the same contract. A
    hand-maintained copy would drift the first time a field is added, and the
    failure mode is silent: the OpenRouter teacher would emit labels missing a
    field that the Gemini teacher always fills, and the difference would surface
    only as unexplained per-field recall loss in the final table.

    Three concrete dialect differences, each of which is a hard error upstream if
    left untranslated:

    * `nullable: true` -> `type: ["x", "null"]`. OpenAI-dialect validators do not
      recognise the OpenAPI `nullable` keyword and will reject a null the model
      legitimately wants to emit -- and null is the *correct* answer for most of
      our fields, so this is the difference that matters most.
    * `additionalProperties: false` is mandatory on every object under
      `strict: true`; omitting it fails schema validation at request time.
    * `propertyOrdering` is Gemini-only and is dropped. Field order is therefore
      not pinned here, which is why `to_target_json` re-serialises in canonical
      order before anything compares two samples byte-wise.
    """
    node_type = node["type"]
    nullable = bool(node.get("nullable", False))
    out: dict[str, Any] = {"type": [node_type, "null"] if nullable else node_type}

    if "enum" in node:
        # `null` goes in the enum list as well as the type union. Per JSON Schema,
        # `enum` constrains *every* instance including nulls, so a nullable enum
        # that omits null is unsatisfiable -- the model is told it may return null
        # and simultaneously that null is not an allowed value. Providers using a
        # real grammar compiler (xgrammar, outlines) enforce that contradiction.
        out["enum"] = [*node["enum"], None] if nullable else list(node["enum"])

    if node_type == "object":
        props = node["properties"]
        out["properties"] = {k: _strict_json_schema(v) for k, v in props.items()}
        out["required"] = list(props)  # strict mode: every property must be required
        out["additionalProperties"] = False
    elif node_type == "array":
        out["items"] = _strict_json_schema(node["items"])

    # Copied by name rather than dropped. These are the keywords that bound
    # runaway generation and keep the grammar in step with pydantic; losing them
    # in translation would leave the OpenRouter path exposed to a failure the
    # Gemini path is protected from, and the symptom (a 23KB reply of one
    # repeated word) looks nothing like a schema-translation bug.
    for keyword in ("maxItems", "maxLength", "minimum", "maximum"):
        if keyword in node:
            out[keyword] = node[keyword]

    return out


def openai_response_schema(name: str = "job_posting") -> dict[str, Any]:
    """`response_format` payload for OpenAI-compatible APIs (OpenRouter, vLLM...).

    Returns the full `response_format` object, not just the schema, because
    `strict: true` is not optional for our purposes: without it most providers
    treat the schema as a *hint* appended to the prompt rather than a decoding
    constraint, and we are back to JSON-repair heuristics in the data path --
    exactly what constrained decoding was chosen to eliminate.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": _strict_json_schema(gemini_response_schema()),
        },
    }


# ---------------------------------------------------------------------------
# Flatten / access helpers shared by eval and the audit
# ---------------------------------------------------------------------------


STRICT_GROUNDED_FIELDS: Final[tuple[str, ...]] = ("company_name", "location.city", "location.region")


def _digits_present(value: float, text: str) -> bool:
    """Is a numeric salary actually supported by the source text?

    Accepts the surface forms a posting really uses for 120000.0:
    "120000", "120,000", "120k", "120 000", and the bare "120" in "$120K".
    """
    n = round(value)
    forms = {str(n), f"{n:,}", f"{n:,}".replace(",", " ")}
    if n % 1000 == 0:
        forms |= {f"{n // 1000}k", f"{n // 1000} k", str(n // 1000)}
    if n % 100 == 0:
        forms.add(f"{n / 1000:g}k")
    low = text.lower()
    return any(f.lower() in low for f in forms)


def ungrounded_fields(obj: JobPosting, source: str) -> list[str]:
    """Names of fields whose value is not supported by `source`.

    Lives in schema.py because it is used at three different stages and must mean
    exactly the same thing at each: it filters teacher labels during data
    generation, it defines the *hallucination rate* metric in eval, and it powers
    the grounding warning in the Gradio app. Three copies of this heuristic would
    guarantee three different hallucination numbers.

    `source` is truncated to the model's actual context first -- see
    truncate_source. Only *extractive* fields are checked; inferred fields
    (seniority_level, employment_type, education_requirement) legitimately have
    no verbatim span, and flagging them would report correct inference as
    hallucination.
    """
    source = truncate_source(source)
    low = source.lower()
    flat = flatten(obj)
    bad: list[str] = []

    for fname in STRICT_GROUNDED_FIELDS:
        val = flat.get(fname)
        # Deliberately substring, not fuzzy: a similarity threshold here would be
        # an unfalsifiable knob that could be tuned until the hallucination number
        # looked good.
        if not (isinstance(val, str) and val.strip()):
            continue
        # location.region is the one field we deliberately REWRITE before this
        # check runs (canonicalize_region maps "Texas" -> "TX"), so a literal
        # substring test asks the source to contain a string the extractor was
        # instructed to replace. A posting that says "Texas" and never "TX"
        # would score as a hallucination for an extraction that is exactly
        # right. Accept either surface form as evidence -- the canonical value
        # and the spelled-out name are the same claim about the world.
        forms = {val.lower()}
        if fname == "location.region":
            spelled = US_STATE_NAMES.get(val.upper())
            if spelled:
                forms.add(spelled)
        if not any(f in low for f in forms):
            bad.append(fname)

    for fname in ("salary.min_amount", "salary.max_amount"):
        val = flat.get(fname)
        if isinstance(val, (int, float)) and not _digits_present(float(val), source):
            bad.append(fname)

    yrs = flat.get("years_experience_min")
    if isinstance(yrs, int) and yrs > 0 and str(yrs) not in low:
        bad.append("years_experience_min")

    dl = flat.get("application_deadline")
    # Only the year is checked: date surface forms vary too much ("Dec 1, 2024"
    # vs "01/12/2024") for a verbatim match to be meaningful.
    if isinstance(dl, str) and dl[:4] not in source:
        bad.append("application_deadline")

    # List fields are checked with a tolerance. Required-skills legitimately gets
    # expanded ("JS" -> "javascript"), so 2 unmatched entries is not evidence of
    # invention. preferred_skills gets zero tolerance because its characteristic
    # failure -- promoting required skills into preferred -- IS the thing we want
    # to catch.
    for fname, tol in (("required_skills", 2), ("preferred_skills", 0), ("benefits", 2)):
        items = flat.get(fname) or []
        misses = sum(1 for s in items if s.split()[0] not in low)
        if misses > tol:
            bad.append(fname)

    return bad


def flatten(obj: JobPosting | dict[str, Any]) -> dict[str, Any]:
    """Flatten to the dotted keys used in FIELD_SPECS.

    Nested objects are scored per-leaf rather than as whole objects. WHY: an
    all-or-nothing `location` score would hide that the model nails city/region
    and fails only on remote_policy, which is exactly the diagnostic a reviewer
    wants. The whole-object view is still available via exact-match rate.
    """
    d = obj.model_dump(mode="json") if isinstance(obj, JobPosting) else obj
    out: dict[str, Any] = {}
    for key in FIELD_SPECS:
        if "." in key:
            parent, child = key.split(".", 1)
            sub = d.get(parent) or {}
            out[key] = sub.get(child) if isinstance(sub, dict) else None
        else:
            out[key] = d.get(key)
    return out


EMPTY_JSON: Final = to_target_json(JobPosting(job_title="unknown"))
