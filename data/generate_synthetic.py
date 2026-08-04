"""Teacher labeling: turn real job postings into verified structured labels.

    python -m data.generate_synthetic --n 5000 --out data/interim/labeled.jsonl
    python -m data.generate_synthetic --n 40 --teacher mock      # no API key needed

Why distillation is a legitimate production pattern (the interview answer)
-------------------------------------------------------------------------
"You used a big model to label data for a small model" is sometimes heard as
cheating. It is not; it is the standard way structured-extraction systems get
built, and the argument has three legs:

1. **It is knowledge transfer, not evaluation leakage.** The frontier model is a
   labeling function, exactly like a crowd worker or a rules engine. What makes
   labels legitimate is verification, not provenance. Every label here passes
   constrained decoding (structurally valid by construction), pydantic
   validation, a *grounding check* against the source text, and -- for val/test
   -- 3-sample self-consistency voting. Labels that fail are dropped, not
   repaired into agreement.

2. **The economics are the point.** Gemini Flash on 100M postings/month is a
   recurring API bill, a network hop, a rate limit and a vendor dependency. A
   1.5B LoRA on a T4 is ~40x cheaper per token, runs in-VPC, and returns in
   ~200ms on CPU as a 1GB GGUF. Distillation converts a per-call cost into a
   one-time training cost. That is why every serious extraction pipeline ends up
   here.

3. **The student can beat the teacher on the deployed task**, because it is
   specialized. The teacher is a generalist paying for capability we do not
   need; the student sees 4k examples of *this* schema and learns the conventions
   (null-over-guess, canonical skill strings, required-vs-preferred) that the
   teacher only follows when the prompt reminds it.

Known limitation, stated up front: test labels come from the teacher, so the
Gemini row in the results table is biased upward. Section
`audit_against_platform` quantifies that bias against non-LLM ground truth, and
the README reports the ceiling comparison as conservative because of it.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.corpus import RawPosting, load_corpus
from data.schema import (
    PLATFORM_AUDITABLE_FIELDS,
    SYSTEM_PROMPT,
    EducationLevel,
    EmploymentType,
    JobPosting,
    Location,
    RemotePolicy,
    Salary,
    SeniorityLevel,
    build_user_prompt,
    canonicalize_terms,
    flatten,
    gemini_response_schema,
    openai_response_schema,
    parse_prediction,
    to_target_json,
    ungrounded_fields,
)

LOGGER = logging.getLogger("teacher")

DEFAULT_MODEL = "gemini-2.0-flash"

#: Default OpenRouter slug. Deliberately the same underlying model family as the
#: Gemini path: if you label part of the corpus through one transport and part
#: through the other, matching the model keeps the label distribution comparable.
#: Switching model families mid-corpus does not -- it puts a systematic
#: annotator difference between the training slice and the held-out slice, which
#: is indistinguishable from the student underfitting.
DEFAULT_OPENROUTER_MODEL = "google/gemini-2.0-flash-exp:free"

#: (requests_per_minute, requests_per_day) per teacher, used when the CLI flags
#: are left unset. These are *starting points*, not guarantees -- free-tier
#: quotas change and OpenRouter's daily cap depends on lifetime credit purchased
#: on the account. The OpenRouter numbers are the conservative (no-credit) case
#: on purpose: overshooting the real cap costs a run of 429s, undershooting only
#: costs one extra day.
DEFAULT_BUDGETS: dict[str, tuple[int, int]] = {
    "gemini": (15, 1500),
    "openrouter": (20, 50),
    "mock": (10_000, 10_000_000),
}


# ===========================================================================
# 1. Rate-limited, disk-cached teacher client
# ===========================================================================


@dataclass
class Budget:
    """Free-tier accounting.

    The Google AI Studio free tier for gemini-2.0-flash allows ~15 requests per
    minute and 1500 requests per day. Labeling 5000 postings (train x1,
    val x1, test x3 for self-consistency) is ~6000 requests -- about four days of
    wall clock. That is a real constraint, not a rounding error, so this pipeline
    is built to be *interrupted and resumed*: every response is cached to disk by
    prompt hash, and a rerun costs zero requests for anything already labeled.

    Split the work across `gemini-2.0-flash` and `gemini-2.0-flash-lite` (each
    carries its own daily quota) with --teacher-model to roughly halve wall clock.
    """

    requests_per_minute: int = 15
    requests_per_day: int = 1500
    _spent: int = 0
    _window_start: float = 0.0
    _window_count: int = 0

    def take(self) -> None:
        if self._spent >= self.requests_per_day:
            raise RuntimeError(
                f"daily budget of {self.requests_per_day} requests exhausted. "
                "Re-run tomorrow -- the cache makes this resumable and free."
            )
        now = time.monotonic()
        if now - self._window_start >= 60.0:
            self._window_start, self._window_count = now, 0
        if self._window_count >= self.requests_per_minute:
            sleep_for = 60.0 - (now - self._window_start) + 0.5
            LOGGER.debug("rpm limit hit, sleeping %.1fs", sleep_for)
            time.sleep(max(0.0, sleep_for))
            self._window_start, self._window_count = time.monotonic(), 0
        self._window_count += 1
        self._spent += 1

    @property
    def spent(self) -> int:
        return self._spent


class ResponseCache:
    """Append-only JSONL cache keyed by (model, temperature, prompt) hash.

    Append-only rather than a dict-dump-on-exit so a Ctrl-C or a 429 storm never
    loses work already paid for out of the daily quota.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, str] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                        self._mem[rec["k"]] = rec["v"]
                    except (json.JSONDecodeError, KeyError):
                        continue  # tolerate a torn last line from a hard kill
            LOGGER.info("teacher cache: %d entries loaded from %s", len(self._mem), self.path)
        self._fh = self.path.open("a", encoding="utf-8")

    @staticmethod
    def key(model: str, temperature: float, prompt: str, sample_idx: int = 0) -> str:
        """Cache key. `sample_idx` is load-bearing, not cosmetic.

        Self-consistency draws k samples at the SAME temperature, and the whole
        point is that they may differ. Without the index in the key, samples 2..k
        collide on (model, temperature, prompt) and every one after the first is
        served from cache as a byte-identical copy.

        The vote then sees k-1 identical objects plus one greedy sample, so the
        duplicated value wins the majority by construction and disagreement
        becomes unobservable -- the exact signal the protocol exists to measure.
        Per-field agreement would read 2/3 on genuinely contested fields and pass
        the 0.67 gate, making the gold set look verified when it is not.
        """
        return hashlib.sha1(
            f"{model}|{temperature:.2f}|{sample_idx}|{prompt}".encode()
        ).hexdigest()

    def get(self, k: str) -> str | None:
        return self._mem.get(k)

    def put(self, k: str, v: str) -> None:
        self._mem[k] = v
        self._fh.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class GeminiTeacher:
    """Thin wrapper over google-genai with constrained decoding.

    `response_schema` + `response_mime_type=application/json` makes the teacher's
    output structurally valid by construction. That is deliberate: it means every
    label we reject later was rejected for a *semantic* reason we chose, never
    because the teacher forgot a brace. It also removes JSON-repair heuristics
    from the data path, which are a classic silent source of label noise.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache: ResponseCache | None = None,
        budget: Budget | None = None,
        api_key: str | None = None,
    ) -> None:
        from google import genai  # local import so --teacher mock needs no SDK

        from config import require_gemini_key

        key = api_key or require_gemini_key()
        self.client = genai.Client(api_key=key)
        self.model = model
        self.cache = cache
        self.budget = budget or Budget()
        self._schema = gemini_response_schema()

    def __call__(self, posting_text: str, temperature: float = 0.0, sample_idx: int = 0) -> str:
        from google.genai import types
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

        prompt = build_user_prompt(posting_text)
        ck = ResponseCache.key(self.model, temperature, prompt, sample_idx)
        if self.cache and (hit := self.cache.get(ck)) is not None:
            return hit

        @retry(
            # 429 (quota) and 503 (overloaded) are the two failure modes on the
            # free tier. Exponential backoff to 60s; beyond 5 attempts the daily
            # quota is genuinely gone and retrying just burns wall clock.
            retry=retry_if_exception_type(Exception),
            wait=wait_exponential(multiplier=2, min=2, max=60),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        def _call() -> str:
            self.budget.take()
            resp = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=self._schema,
                    temperature=temperature,
                    max_output_tokens=1024,
                ),
            )
            return resp.text or ""

        out = _call()
        if self.cache:
            self.cache.put(ck, out)
        return out


class OpenRouterTeacher:
    """Same protocol as GeminiTeacher, against any OpenAI-compatible model.

    Why this exists: it decouples the labeling protocol from one vendor's free
    tier. The self-consistency vote, the grounding filter and the platform audit
    are all teacher-agnostic; only the transport was Google-specific. With this
    class the teacher becomes a swappable component, which is also the honest
    answer to "what if Gemini's free tier changes?" -- rerun with a different
    `--teacher-model` and the cache, budget and audit all still apply.

    Two things to know before choosing this over Gemini for a bulk run:

    * **Quota, not quality, is the deciding factor.** Google AI Studio gives
      1500 requests/day for free. OpenRouter's free-model allowance is a
      per-account daily cap that is substantially smaller unless credit has been
      purchased. Check yours at https://openrouter.ai/docs/api-reference/limits
      and pass the real number as --requests-per-day; the Budget class will stop
      cleanly at that ceiling instead of collecting a run of 429s.
    * **Structured output is provider-dependent.** Not every backend behind a
      given model slug implements `response_format: json_schema`. We send
      `provider.require_parameters=true` so OpenRouter routes *only* to providers
      that honour it -- without that flag the request silently succeeds against a
      provider that ignored the schema, and the label set quietly degrades to
      whatever free-form JSON the model felt like emitting.
    """

    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model: str = DEFAULT_OPENROUTER_MODEL,
        cache: ResponseCache | None = None,
        budget: Budget | None = None,
        api_key: str | None = None,
        structured: bool = True,
    ) -> None:
        import requests  # local import, mirroring GeminiTeacher's lazy SDK import

        from config import require_openrouter_key

        self._requests = requests
        self.model = model
        self.cache = cache
        self.budget = budget or Budget(requests_per_minute=20, requests_per_day=50)
        self.structured = structured
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key or require_openrouter_key()}",
            "Content-Type": "application/json",
            # OpenRouter attributes traffic by these; harmless but conventional.
            "HTTP-Referer": "https://github.com/2005-Aneeshdutt/structured-extract",
            "X-Title": "structured-extract",
        })
        self._response_format = openai_response_schema() if structured else None
        if not structured:
            LOGGER.warning(
                "structured output DISABLED -- teacher JSON is no longer valid by "
                "construction and malformed replies will be dropped as parse failures"
            )

    def _payload(self, prompt: str, temperature: float) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 1024,
        }
        if self._response_format is not None:
            body["response_format"] = self._response_format
            # Refuse a provider that would ignore response_format rather than
            # accept labels generated without the constraint. See class docstring.
            body["provider"] = {"require_parameters": True}
        return body

    def __call__(self, posting_text: str, temperature: float = 0.0, sample_idx: int = 0) -> str:
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

        prompt = build_user_prompt(posting_text)
        ck = ResponseCache.key(self.model, temperature, prompt, sample_idx)
        if self.cache and (hit := self.cache.get(ck)) is not None:
            return hit

        @retry(
            retry=retry_if_exception_type(Exception),
            wait=wait_exponential(multiplier=2, min=2, max=60),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        def _call() -> str:
            self.budget.take()
            resp = self._session.post(
                self.URL, json=self._payload(prompt, temperature), timeout=120
            )
            if resp.status_code != 200:
                raise RuntimeError(f"openrouter HTTP {resp.status_code}: {resp.text[:400]}")
            data = resp.json()
            # OpenRouter reports upstream provider failures inside a 200 body.
            # Treating that as success would write the error object into the cache
            # as if it were a label, and it would only surface later as an
            # unparseable record with no trace of what went wrong.
            if "error" in data:
                raise RuntimeError(f"openrouter error: {json.dumps(data['error'])[:400]}")
            try:
                return data["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError) as exc:
                raise RuntimeError(f"unexpected response shape: {json.dumps(data)[:400]}") from exc

        out = _call()
        if self.cache:
            self.cache.put(ck, out)
        return out


# ===========================================================================
# 2. Mock teacher -- lets CI and a laptop run the whole pipeline with no key
# ===========================================================================

_MONEY_RE = re.compile(
    r"(?P<cur>[$£€₹]|\b(?:USD|GBP|EUR|INR)\b)?\s?(?P<a>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(?P<k>k\b)?",
    re.IGNORECASE,
)
_CUR_MAP = {"$": "USD", "£": "GBP", "€": "EUR", "₹": "INR"}
_YEARS_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:-|to|–)?\s*(?:\d{1,2})?\s*years?", re.IGNORECASE)
_SKILL_VOCAB = ["python", "java", "javascript", "typescript", "sql", "excel", "aws", "azure", "react", "node.js", "docker", "kubernetes", "git", "linux", "c#", "c++", "go", "rust", "scala", "spark", "hadoop", "tableau", "power", "bi", "salesforce", "sap", "communication", "leadership", "project", "management"]


def mock_teacher(posting_text: str, temperature: float = 0.0, sample_idx: int = 0) -> str:
    """Deterministic regex 'teacher' for offline testing.

    NOT used for real labels -- its outputs are far too crude. It exists so that
    `pytest`, CI and a first-run smoke test can exercise every downstream stage
    (validation, voting, audit, splitting, SFT formatting) without an API key or
    a four-day quota spend. Keeping the interface identical to GeminiTeacher
    means the code path under test is the real one.
    """
    low = posting_text.lower()
    m = _MONEY_RE.search(posting_text)
    salary = Salary()
    if m and (m.group("cur") or m.group("k")):
        amt = float(m.group("a").replace(",", ""))
        if m.group("k"):
            amt *= 1000
        sym = (m.group("cur") or "$").upper()
        if 0 < amt < 1e9:
            salary = Salary(
                min_amount=amt,
                max_amount=amt,
                currency=_CUR_MAP.get(sym, sym if len(sym) == 3 else "USD"),
                period="hourly" if amt < 200 else "yearly",
            )
    ym = _YEARS_RE.search(posting_text)
    obj = JobPosting(
        job_title=posting_text.split("\n", 1)[0][:80].strip() or "unknown",
        company_name=None,
        location=Location(
            remote_policy=RemotePolicy.REMOTE if "remote" in low else (RemotePolicy.HYBRID if "hybrid" in low else None)
        ),
        employment_type=EmploymentType.FULL_TIME if "full-time" in low or "full time" in low else None,
        seniority_level=SeniorityLevel.SENIOR if "senior" in low else (SeniorityLevel.ENTRY if "entry" in low else None),
        education_requirement=EducationLevel.BACHELOR if "bachelor" in low else None,
        salary=salary,
        years_experience_min=int(ym.group(1)) if ym and int(ym.group(1)) <= 50 else None,
        application_deadline=None,
        required_skills=canonicalize_terms([s for s in _SKILL_VOCAB if s in low], limit=15),
        preferred_skills=[],
        benefits=canonicalize_terms([b for b in ("health insurance", "401k", "pto", "remote work") if b in low], limit=10),
    )
    return to_target_json(obj)


# ===========================================================================
# 3. Grounding verification -- the step that makes the labels trustworthy
# ===========================================================================

#: Grounding verification lives in `schema.ungrounded_fields` so that the label
#: filter here and the hallucination *metric* in eval/ are literally the same
#: function. Constrained decoding guarantees the teacher's output shape; nothing
#: guarantees its truth. Without this filter the student would be trained on the
#: teacher's occasional inventions -- and since reduced hallucination is one of
#: our headline claims, training on hallucinated labels would quietly invalidate
#: the result.
grounding_failures = ungrounded_fields


# ===========================================================================
# 4. Self-consistency voting (val/test only)
# ===========================================================================


def _vote_value(values: list[Any]) -> tuple[Any, float]:
    """Majority vote over one leaf field; returns (winner, agreement in [0,1])."""
    keyed = [json.dumps(v, sort_keys=True, default=str) for v in values]
    top, count = Counter(keyed).most_common(1)[0]
    return json.loads(top), count / len(keyed)


def vote(samples: list[JobPosting]) -> tuple[JobPosting | None, dict[str, float]]:
    """Per-leaf-field majority vote across k teacher samples.

    WHY per-field rather than per-object: with 18 leaf fields, whole-object
    agreement across 3 samples is rare (~30%), so object-level voting would throw
    away most of the data. Per-field voting keeps the examples and gives a
    *per-field confidence* we can filter on -- fields where the teacher disagrees
    with itself are exactly the ambiguous ones we do not want in a held-out test
    set claiming to be ground truth.

    Returns (voted_object, per_field_agreement). Object is None if the vote
    produces something that no longer validates.
    """
    if not samples:
        return None, {}
    flats = [flatten(s) for s in samples]
    voted: dict[str, Any] = {}
    agreement: dict[str, float] = {}
    for fname in flats[0]:
        val, agr = _vote_value([f[fname] for f in flats])
        voted[fname] = val
        agreement[fname] = agr

    nested: dict[str, Any] = {"location": {}, "salary": {}}
    payload: dict[str, Any] = {}
    for fname, val in voted.items():
        if "." in fname:
            parent, child = fname.split(".", 1)
            nested[parent][child] = val
        else:
            payload[fname] = val
    payload.update(nested)
    try:
        return JobPosting.model_validate(payload), agreement
    except Exception as e:  # a vote can mix incompatible sub-values
        LOGGER.debug("vote produced invalid object: %s", e)
        return None, agreement


# ===========================================================================
# 5. Platform-metadata audit -- independent, non-LLM label quality evidence
# ===========================================================================

_WORKTYPE_MAP = {
    "full-time": "full_time", "part-time": "part_time", "contract": "contract",
    "internship": "internship", "temporary": "temporary", "volunteer": "volunteer",
    "other": None,
}
_EXPLEVEL_MAP = {
    "internship": "intern", "entry level": "entry", "associate": "mid",
    "mid-senior level": "senior", "director": "lead", "executive": "executive",
}
_PERIOD_MAP = {"HOURLY": "hourly", "WEEKLY": "weekly", "MONTHLY": "monthly", "YEARLY": "yearly"}


def audit_against_platform(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compare teacher labels to LinkedIn's own structured fields.

    This is the answer to "how do you know your labels are any good, given a
    model produced them?" -- for 10 of 18 leaf fields there is independent,
    non-LLM ground truth from the posting's platform metadata.

    Read it as a PRECISION check, not recall. The platform fields come from the
    employer's form, which frequently records a salary the description never
    mentions; in that case the correct label is null and a teacher "miss" is
    correct behavior. So we score only the cases where the teacher *did* emit a
    value, and ask whether it agrees. Disagreement there is unambiguously a
    teacher error.
    """
    stats: dict[str, dict[str, Any]] = {}

    def bump(field_name: str, ok: bool) -> None:
        s = stats.setdefault(field_name, {"n_compared": 0, "n_agree": 0})
        s["n_compared"] += 1
        s["n_agree"] += int(ok)

    for rec in records:
        plat = rec.get("platform") or {}
        flat = flatten(json.loads(rec["target_json"]))

        if (t := flat.get("job_title")) and plat.get("title"):
            # Token overlap, not equality: our schema asks for a cleaned title
            # ("Senior Engineer") while LinkedIn stores the raw advert string
            # ("Senior Engineer - Remote - REQ12345"). Containment either way is
            # agreement.
            a, b = t.lower(), str(plat["title"]).lower()
            bump("job_title", a in b or b in a or len(set(a.split()) & set(b.split())) >= 2)

        loc = str(plat.get("location") or "")
        if (c := flat.get("location.city")) and loc:
            bump("location.city", c.lower() in loc.lower())
        if (r := flat.get("location.region")) and loc:
            bump("location.region", r.lower() in loc.lower())

        if (et := flat.get("employment_type")) and plat.get("formatted_work_type"):
            bump("employment_type", _WORKTYPE_MAP.get(str(plat["formatted_work_type"]).lower()) == et)

        if (sl := flat.get("seniority_level")) and plat.get("formatted_experience_level"):
            expected = _EXPLEVEL_MAP.get(str(plat["formatted_experience_level"]).lower())
            # Adjacent levels are not an error -- "Mid-Senior level" genuinely
            # spans mid and senior. Only distant disagreement counts.
            order = ["intern", "entry", "mid", "senior", "lead", "principal", "executive"]
            bump("seniority_level", expected is not None and abs(order.index(sl) - order.index(expected)) <= 1)

        if flat.get("location.remote_policy") and plat.get("remote_allowed") is not None:
            is_remote = flat["location.remote_policy"] in ("remote", "hybrid")
            bump("location.remote_policy", bool(plat["remote_allowed"]) == is_remote)

        for our, theirs in (("salary.min_amount", "min_salary"), ("salary.max_amount", "max_salary")):
            v, p = flat.get(our), plat.get(theirs)
            if v is not None and p not in (None, ""):
                # A non-numeric platform value just means this posting cannot be
                # audited on this field -- skip it rather than counting it as a
                # disagreement, which would understate label quality.
                with contextlib.suppress(TypeError, ValueError):
                    # 5% tolerance absorbs rounding ("~$120k" vs 119,500).
                    bump(our, abs(float(v) - float(p)) <= 0.05 * max(float(p), 1.0))

        if (cur := flat.get("salary.currency")) and plat.get("currency"):
            bump("salary.currency", cur == str(plat["currency"]).upper())
        if (per := flat.get("salary.period")) and plat.get("pay_period"):
            bump("salary.period", _PERIOD_MAP.get(str(plat["pay_period"]).upper()) == per)

    for s in stats.values():
        s["agreement"] = round(s["n_agree"] / s["n_compared"], 4) if s["n_compared"] else None
    return {k: stats[k] for k in PLATFORM_AUDITABLE_FIELDS if k in stats}


# ===========================================================================
# 6. Driver
# ===========================================================================

TeacherFn = Callable[..., str]


def label_posting(
    posting: RawPosting,
    teacher: TeacherFn,
    *,
    n_samples: int,
    temperature: float,
) -> dict[str, Any]:
    """Label one posting. Returns a record with a `status` field, never raises.

    Statuses form the funnel reported at the end of the run:
      ok | parse_failed | ungrounded | vote_failed | low_agreement
    Keeping rejected records (rather than dropping silently) is what lets us
    report a yield number and inspect *why* labels were lost -- a silent filter
    is how label bias sneaks into a dataset.
    """
    base = {"posting_id": posting.posting_id, "source_text": posting.text, "platform": posting.platform}

    # Sample 0 is greedy; samples 1..k-1 share a temperature but are distinct
    # draws, so each carries its own sample_idx to keep the cache from collapsing
    # them into one repeated answer. See ResponseCache.key.
    raws = [
        teacher(posting.text, temperature=temperature if i else 0.0, sample_idx=i)
        for i in range(n_samples)
    ]
    parsed: list[JobPosting] = []
    for raw in raws:
        obj, _err = parse_prediction(raw)
        if obj is not None:
            parsed.append(obj)
    if not parsed:
        return {**base, "status": "parse_failed"}

    if n_samples > 1:
        obj, agreement = vote(parsed)
        if obj is None:
            return {**base, "status": "vote_failed"}
        # Gate on the hard fields only. Requiring unanimity everywhere would
        # reject ~half the corpus over cosmetic city/region disagreements while
        # doing nothing for the fields that actually carry ambiguity.
        hard = ["salary.min_amount", "salary.max_amount", "salary.period",
                "years_experience_min", "application_deadline", "seniority_level"]
        min_agr = min(agreement.get(f, 1.0) for f in hard)
        if min_agr < 0.67:
            return {**base, "status": "low_agreement", "agreement": agreement}
    else:
        obj, agreement = parsed[0], {}

    bad = grounding_failures(obj, posting.text)
    if bad:
        return {**base, "status": "ungrounded", "ungrounded_fields": bad}

    return {
        **base,
        "status": "ok",
        "target_json": to_target_json(obj),
        "agreement": agreement,
        "n_samples": n_samples,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5000, help="postings to pull from the corpus")
    ap.add_argument("--max-corpus-rows", type=int, default=None,
                    help="cap rows scanned from the source dataset (smoke tests / CI only)")
    ap.add_argument("--out", type=Path, default=Path("data/interim/labeled.jsonl"))
    ap.add_argument("--cache", type=Path, default=Path("data/interim/teacher_cache.jsonl"))
    ap.add_argument("--teacher", choices=["gemini", "openrouter", "mock"], default="gemini")
    ap.add_argument("--teacher-model", default=None,
                    help="model id; defaults to the right one for --teacher")
    ap.add_argument("--no-structured-output", action="store_true",
                    help="openrouter only: drop the json_schema constraint. Widens model "
                         "choice at the cost of guaranteed-valid JSON -- expect parse losses")
    ap.add_argument("--n-samples", type=int, default=1,
                    help="teacher samples per posting; use 3 for the val/test slice (self-consistency)")
    ap.add_argument("--temperature", type=float, default=0.4,
                    help="temperature for samples 2..k; sample 1 is always greedy")
    # Default None, not a number: the right ceiling depends on --teacher, and a
    # hardcoded 1500 silently invites 1450 failed requests against a 50/day cap.
    ap.add_argument("--requests-per-day", type=int, default=None,
                    help="daily ceiling; defaults per --teacher. Set to YOUR real quota")
    ap.add_argument("--requests-per-minute", type=int, default=None,
                    help="per-minute ceiling; defaults per --teacher")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--exclude-from", type=Path, action="append", default=[],
                    help="skip posting_ids already labeled in this file; repeatable")
    ap.add_argument("--audit-out", type=Path, default=Path("results/label_audit.md"))
    args = ap.parse_args(argv)

    # Validate arguments BEFORE the corpus load. A flag that silently does
    # nothing is how a run ends up believing it tested the unconstrained path
    # when it did not -- and rejecting it down at teacher construction would
    # charge the user a multi-minute dataset fetch to learn about a typo.
    if args.no_structured_output and args.teacher != "openrouter":
        ap.error("--no-structured-output applies only to --teacher openrouter")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Before load_corpus: HF_TOKEN must be in os.environ prior to the Hub client
    # initializing, or the download runs anonymously despite a valid token.
    from config import setup_run

    setup_run()
    random.seed(args.seed)

    postings = load_corpus(args.n, seed=args.seed, max_rows=args.max_corpus_rows)
    LOGGER.info("labeling %d postings with teacher=%s n_samples=%d", len(postings), args.teacher, args.n_samples)

    cache = ResponseCache(args.cache)
    # Budget precedence: CLI flag > .env > per-teacher default. The .env layer
    # exists because the ceiling is a property of YOUR account, not of the
    # command being run -- setting it once beats remembering a flag on every
    # invocation of a run that spans days.
    from config import get_int

    default_rpm, default_rpd = DEFAULT_BUDGETS[args.teacher]
    budget = Budget(
        requests_per_minute=args.requests_per_minute or get_int("REQUESTS_PER_MINUTE") or default_rpm,
        requests_per_day=args.requests_per_day or get_int("REQUESTS_PER_DAY") or default_rpd,
    )
    teacher: TeacherFn
    teacher_model = args.teacher_model or {
        "gemini": DEFAULT_MODEL,
        "openrouter": DEFAULT_OPENROUTER_MODEL,
        "mock": "mock",
    }[args.teacher]
    LOGGER.info(
        "teacher=%s model=%s budget=%d/min %d/day",
        args.teacher, teacher_model, budget.requests_per_minute, budget.requests_per_day,
    )
    if args.teacher == "mock":
        teacher = mock_teacher
    elif args.teacher == "openrouter":
        teacher = OpenRouterTeacher(
            model=teacher_model,
            cache=cache,
            budget=budget,
            structured=not args.no_structured_output,
        )
    else:
        teacher = GeminiTeacher(model=teacher_model, cache=cache, budget=budget)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Resume support: skip postings already written on a previous (quota-limited)
    # run. Combined with the response cache this makes the multi-day labeling run
    # fully restartable.
    def _ids_in(path: Path) -> set[str]:
        ids: set[str] = set()
        if not path.exists():
            return ids
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    ids.add(json.loads(line)["posting_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        return ids

    done: set[str] = _ids_in(args.out)
    if done:
        LOGGER.info("resuming: %d postings already labeled in %s", len(done), args.out)

    # --exclude-from makes the two-phase protocol possible. Held-out gold is
    # labeled first with 3-sample self-consistency; the bulk training pass then
    # draws from a LARGER corpus slice that is a superset of the gold slice, so
    # without this the same postings would be labeled twice -- burning quota and,
    # worse, putting duplicate posting_ids into prepare_dataset where they could
    # straddle the train/test boundary.
    for path in args.exclude_from:
        excluded = _ids_in(path)
        done |= excluded
        LOGGER.info("excluding %d posting_ids already labeled in %s", len(excluded), path)

    statuses: Counter[str] = Counter()
    ok_records: list[dict[str, Any]] = []
    with args.out.open("a", encoding="utf-8") as fh:
        for i, p in enumerate(postings, 1):
            if p.posting_id in done:
                continue
            try:
                rec = label_posting(p, teacher, n_samples=args.n_samples, temperature=args.temperature)
            except RuntimeError as e:  # daily budget exhausted -- stop cleanly
                LOGGER.warning("stopping early: %s", e)
                break
            statuses[rec["status"]] += 1
            if rec["status"] == "ok":
                ok_records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                LOGGER.info("%d/%d | yield=%.1f%% | %s", i, len(postings),
                            100 * statuses["ok"] / max(sum(statuses.values()), 1), dict(statuses))
    cache.close()

    total = sum(statuses.values())
    LOGGER.info("done. %d processed, yield %.1f%%, funnel=%s",
                total, 100 * statuses["ok"] / max(total, 1), dict(statuses))

    if ok_records:
        audit = audit_against_platform(ok_records)
        args.audit_out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Teacher label audit vs. platform metadata",
            "",
            f"Teacher: `{teacher_model}` via `{args.teacher}`  |  "
            f"labeled examples audited: {len(ok_records)}",
            "",
            "Agreement between teacher labels and LinkedIn's own structured form fields "
            "(independent of any LLM). Scored only where the teacher emitted a non-null "
            "value, so this measures label **precision**; a null where the platform has a "
            "value is usually correct, because the form field is often absent from the "
            "description text.",
            "",
            "| field | n compared | agreement |",
            "|---|---:|---:|",
        ]
        for fname, s in audit.items():
            lines.append(f"| `{fname}` | {s['n_compared']} | {s['agreement']:.1%} |" if s["agreement"] is not None
                         else f"| `{fname}` | 0 | n/a |")
        lines += ["", "## Label funnel", "", "| status | count | share |", "|---|---:|---:|"]
        for st, c in statuses.most_common():
            lines.append(f"| `{st}` | {c} | {c / max(total, 1):.1%} |")
        args.audit_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        LOGGER.info("wrote audit -> %s", args.audit_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
