"""Real-world job-posting corpus: fetch, clean, deduplicate, stratify.

Source: `xanderios/linkedin-job-postings` on the Hugging Face Hub -- 33,246 real
LinkedIn postings scraped in 2023-2024, MIT licensed, no auth required.

WHY a real corpus instead of fully synthetic text
-------------------------------------------------
The alternative (ask Gemini to *write* job postings, then label them) produces a
closed loop: the same model invents the text and the answer, so the text is
always well-formed, always states the salary in the same phrasing, and the
student learns the teacher's writing style rather than the extraction task.
Every number would then be measured on a distribution that does not exist in
production. Here only the *labels* are model-generated; the inputs are real
postings with real noise -- inconsistent headers, missing spaces where HTML tags
were stripped, mojibake, 6 KB of EEO boilerplate. That is the distribution the
model must survive.

The second, larger reason: this corpus ships LinkedIn's own structured form
fields (`min_salary`, `pay_period`, `formatted_work_type`,
`formatted_experience_level`, `location`, `remote_allowed`). Those are *human
and platform generated, not LLM generated*, and they let us audit teacher labels
against independent ground truth on 10 of our 18 scored leaf fields. See
`generate_synthetic.py::audit_against_platform`.
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import ftfy

LOGGER = logging.getLogger(__name__)

HF_DATASET_ID = "xanderios/linkedin-job-postings"

# Length gates, in characters, applied to the *cleaned* text.
MIN_CHARS = 400    # below this the posting is a stub with nothing to extract
MAX_CHARS = 20_000  # above this it is a company-wide careers dump, not one role

#: Columns carried through as independent (non-LLM) ground truth for the audit.
#: `company_id` is not ground truth -- it is carried so prepare_dataset.py can do
#: a GROUPED train/val/test split (no company spans two splits). Employers reuse
#: boilerplate across their own postings, so a random split would leak style and
#: benefits phrasing across the boundary even after near-dup removal.
PLATFORM_COLUMNS = (
    "company_id",
    "title",
    "location",
    "min_salary",
    "max_salary",
    "med_salary",
    "pay_period",
    "currency",
    "formatted_work_type",
    "formatted_experience_level",
    "remote_allowed",
)


@dataclass(slots=True)
class RawPosting:
    """One cleaned posting plus the platform metadata used only for auditing.

    `platform` is NEVER shown to any model. It exists solely so we can ask
    "when the teacher extracted a salary, did it match the number LinkedIn's own
    form recorded?" -- keeping it out of the prompt is what makes that check
    independent.
    """

    posting_id: str
    text: str
    platform: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha1(self.text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

_BULLET_RE = re.compile(r"[·•‣▪]\s*")
_WS_RE = re.compile(r"[ \t ]+")
_NL_RE = re.compile(r"\n{3,}")
_URL_RE = re.compile(r"https?://\S+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")


def clean_text(raw: str) -> str:
    """Normalize encoding and whitespace. Deliberately light-touch.

    What we DO fix:
      * mojibake ("Â·", "â€™") -- an encoding bug introduced by the scraper, not
        a property of the real distribution. Leaving it in would teach the model
        to expect broken UTF-8.
      * bullet glyphs collapsed into the previous word -- restored to newlines,
        because section boundaries carry the required-vs-preferred signal that
        two of our fields depend on.
      * emails and URLs -> placeholders. PII hygiene, and they are pure noise
        tokens that would otherwise eat the 6000-char budget.

    What we deliberately do NOT fix: missing spaces at former tag boundaries
    ("Job DescriptionThe Service Desk Technician role..."). That artifact is
    present in the real input a deployed parser receives. Over-cleaning here
    would make the eval optimistic -- the classic mistake of measuring on a
    distribution you manufactured.
    """
    t = ftfy.fix_text(raw or "")
    t = _URL_RE.sub("[URL]", t)
    t = _EMAIL_RE.sub("[EMAIL]", t)
    t = _BULLET_RE.sub("\n- ", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _WS_RE.sub(" ", t)
    t = _NL_RE.sub("\n\n", t)
    return t.strip()


_EN_STOPWORDS = frozenset(
    ["the", "and", "for", "with", "you", "our", "will", "are", "that", "this", "have", "from", "your", "work", "team", "role", "experience", "skills", "we", "job"]
)


def looks_english(text: str, min_hits: int = 8) -> bool:
    """Cheap English filter -- no langdetect dependency.

    A langdetect/fastText model would be more accurate, but this corpus is ~99%
    English already; the filter only needs to catch the occasional Spanish or
    French posting. Counting distinct common English function words is enough,
    costs nothing, and has no model download.
    """
    words = set(re.findall(r"[a-z']+", text.lower()[:4000]))
    return len(words & _EN_STOPWORDS) >= min_hits


# ---------------------------------------------------------------------------
# Stratification signals
# ---------------------------------------------------------------------------

#: Matches "$120,000", "120k", "£45,000", "USD 90,000", "$28.50/hr".
SALARY_IN_TEXT_RE = re.compile(
    r"(?:[$£€₹]\s?\d[\d,.]*\s*(?:k\b)?)"
    r"|(?:\b(?:usd|gbp|eur|cad|inr)\s?\d[\d,.]*)"
    r"|(?:\b\d{2,3}\s?k\b\s*(?:-|to|–)\s*\d{2,3}\s?k\b)",
    re.IGNORECASE,
)

#: A deadline CUE ("apply by", "closing date") followed within 60 chars by
#: something date-shaped. The cue alone is not enough: "apply before the position
#: closes" matches a cue-only regex but contains no extractable date, so a
#: cue-only stratifier would enrich for postings whose correct label is still
#: null -- the opposite of what oversampling is for.
_DEADLINE_RE = re.compile(
    r"\b(?:apply|applications?|deadline|closes?|closing date|submit)\b[^.\n]{0,60}?"
    r"(?:\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)


def stratum_of(posting: RawPosting) -> str:
    """Assign a posting to a sampling stratum.

    WHY stratify at all: measured over the full deduplicated pool, 62% of
    postings mention no currency amount and 98.7% carry no date-shaped deadline.
    Sample uniformly and the model learns the constant function `null` for the
    two hardest fields, while per-field accuracy still *looks* excellent --
    because predicting null is right 98.7% of the time on deadlines. The metric
    would be measuring the base rate, not the model.

    So salary-bearing and deadline-bearing postings are oversampled (see
    STRATUM_TARGETS for the measured availability that sets the ceiling on how
    far that can go). This is a *documented distribution shift*: reported
    salary-field F1 is conditional on this enriched mix and is NOT an estimate of
    wild-distribution performance. Train, val and test are all drawn from the same
    enriched mix, so the model-vs-model comparison -- the claim we actually make --
    remains valid.
    """
    has_salary = bool(SALARY_IN_TEXT_RE.search(posting.text))
    has_deadline = bool(_DEADLINE_RE.search(posting.text))
    if has_salary and has_deadline:
        return "salary+deadline"
    if has_salary:
        return "salary"
    if has_deadline:
        return "deadline"
    return "plain"


#: Target share of the final corpus per stratum, set from MEASURED availability
#: over the full 25,186-posting deduplicated pool:
#:
#:     salary            9,561   (38.0%)
#:     plain            15,312   (60.8%)
#:     salary+deadline     192   ( 0.8%)
#:     deadline            121   ( 0.5%)
#:
#: Salary is enriched from 38% to 40% -- barely a shift, because salary-bearing
#: postings turned out to be plentiful. Explicit deadlines are the opposite: only
#: **313 postings in the entire corpus** carry a date-shaped deadline, so the two
#: deadline strata are set to take essentially all of them. Asking for more would
#: silently under-deliver.
#:
#: Consequence, stated so it is not mistaken for an oversight: at n=5,000 the
#: `application_deadline` field has roughly 300 positives overall and ~30 in the
#: test split. Its F1 confidence interval is therefore wide and should not carry
#: an argument alone. Its real diagnostic value is on the *negative* class --
#: ~470 test postings where the correct answer is null and a base model invents a
#: date anyway. That shows up in over_emission_rate and null_recall, which have
#: ample support.
STRATUM_TARGETS: dict[str, float] = {
    "salary+deadline": 0.038,
    "deadline": 0.024,
    "salary": 0.40,
    "plain": 0.538,
}


# ---------------------------------------------------------------------------
# Near-duplicate removal
# ---------------------------------------------------------------------------


def _shingles(text: str, k: int = 5) -> set[bytes]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < k:
        return {" ".join(words).encode()}
    return {" ".join(words[i : i + k]).encode() for i in range(len(words) - k + 1)}


def deduplicate(postings: list[RawPosting], threshold: float = 0.75) -> list[RawPosting]:
    """Exact + near-duplicate removal via MinHash LSH.

    THIS IS THE MOST IMPORTANT FUNCTION IN THE DATA PIPELINE.

    Large employers post the same requisition across 40 cities with only the
    location line changed. Jaccard similarity on 5-word shingles between two such
    postings is typically 0.95+. If deduplication happens after the train/test
    split -- or not at all -- near-identical postings land on both sides and the
    test set stops being held out. Every headline number would then be partly
    memorization, and that is the first thing a senior reviewer probes.

    Order of operations, enforced by prepare_dataset.py: dedupe -> split. Never
    the reverse.

    threshold=0.75 is deliberately aggressive. Cost of a false positive is one
    discarded posting out of 33k (free). Cost of a false negative is a
    contaminated test set (fatal). Asymmetric loss -> asymmetric threshold.
    """
    from datasketch import MinHash, MinHashLSH  # local import: heavy, optional

    seen_exact: set[str] = set()
    unique: list[RawPosting] = []
    lsh = MinHashLSH(threshold=threshold, num_perm=128)

    n_exact = n_near = 0
    for p in postings:
        h = p.content_hash
        if h in seen_exact:
            n_exact += 1
            continue
        seen_exact.add(h)

        m = MinHash(num_perm=128)
        for sh in _shingles(p.text):
            m.update(sh)
        if lsh.query(m):
            n_near += 1
            continue
        lsh.insert(p.posting_id, m)
        unique.append(p)

    LOGGER.info(
        "dedup: %d in -> %d out (%d exact, %d near-dup removed)",
        len(postings), len(unique), n_exact, n_near,
    )
    return unique


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _to_posting(row: dict[str, Any]) -> RawPosting | None:
    text = clean_text(row.get("description") or "")
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return None
    if not looks_english(text):
        return None
    return RawPosting(
        posting_id=str(row.get("job_id") or hashlib.sha1(text.encode()).hexdigest()[:16]),
        text=text,
        platform={c: row.get(c) for c in PLATFORM_COLUMNS},
    )


def load_corpus(
    n_target: int,
    *,
    seed: int = 13,
    dataset_id: str = HF_DATASET_ID,
    dedup_threshold: float = 0.75,
    stratum_targets: dict[str, float] | None = None,
    max_rows: int | None = None,
) -> list[RawPosting]:
    """Load, clean, dedupe and stratify down to `n_target` postings.

    Pipeline order is load -> clean -> filter -> DEDUPE -> stratify -> sample.
    Deduplication happens before stratified sampling so that duplicate-heavy
    strata (large employers post salary bands most consistently) cannot dominate.
    """
    from datasets import load_dataset  # local import keeps schema.py-only users light

    LOGGER.info("loading %s ...", dataset_id)
    ds = load_dataset(dataset_id, split="train")

    n_seen = 0
    postings: list[RawPosting] = []
    for row in ds:
        n_seen += 1
        p = _to_posting(row)
        if p is not None:
            postings.append(p)
        # max_rows exists for smoke tests and CI only. Production runs scan the
        # whole corpus, because the stratified sampler needs the full pool to
        # find enough salary-bearing postings.
        if max_rows is not None and n_seen >= max_rows:
            break
    LOGGER.info("cleaned+filtered: %d / %d rows survived", len(postings), n_seen)

    rng = random.Random(seed)
    rng.shuffle(postings)  # shuffle BEFORE dedup so LSH keeps a random representative
    postings = deduplicate(postings, threshold=dedup_threshold)

    targets = stratum_targets or STRATUM_TARGETS
    buckets: dict[str, list[RawPosting]] = {k: [] for k in targets}
    for p in postings:
        buckets.setdefault(stratum_of(p), []).append(p)

    selected: list[RawPosting] = []
    for name, share in targets.items():
        want = round(n_target * share)
        pool = buckets.get(name, [])
        if len(pool) < want:
            LOGGER.warning("stratum %r: wanted %d, only %d available", name, want, len(pool))
        selected.extend(pool[:want])

    # Backfill from the largest remaining pool if a stratum came up short, so we
    # still hit n_target rather than silently shrinking the dataset.
    if len(selected) < n_target:
        taken = {p.posting_id for p in selected}
        leftovers = [p for p in postings if p.posting_id not in taken]
        selected.extend(leftovers[: n_target - len(selected)])

    rng.shuffle(selected)
    LOGGER.info(
        "corpus ready: %d postings | strata=%s",
        len(selected),
        {k: sum(1 for p in selected if stratum_of(p) == k) for k in targets},
    )
    return selected[:n_target]


def describe_lengths(postings: Iterable[RawPosting]) -> dict[str, float]:
    """Length percentiles -- justifies MAX_SOURCE_CHARS in schema.py."""
    lens = sorted(len(p.text) for p in postings)
    if not lens:
        return {}
    pct = lambda q: float(lens[min(len(lens) - 1, int(q * len(lens)))])  # noqa: E731
    return {"n": len(lens), "p50": pct(0.50), "p90": pct(0.90), "p95": pct(0.95), "max": float(lens[-1])}
