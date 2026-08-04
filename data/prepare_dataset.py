"""Quality-filter, split, and format labeled examples for SFT.

    python -m data.prepare_dataset --in data/interim/labeled.jsonl --out data/processed

Produces, under --out:
    train.jsonl / val.jsonl / test.jsonl   one record per line, human-inspectable
    hf/                                    DatasetDict.save_to_disk() for training
    fewshot_exemplars.json                 3 exemplars for the base-model 3-shot arm
    ../results/dataset_stats.md            per-split stats + the leakage report

Design commitments enforced here
--------------------------------
* **The test split is sacred.** It is written once, its ids are recorded in
  `test_ids.txt`, and nothing downstream of this script may read its labels
  except the eval harness. Model selection uses val only.
* **Split before you look.** Splitting happens on a hash of `posting_id`, so
  adding more labeled data later does not reshuffle existing assignments and
  silently move yesterday's test example into today's train set.
* **Grouped by company.** No `company_id` appears in more than one split.
* **Leakage is verified, not assumed.** A cross-split near-duplicate scan runs
  after splitting and writes its result into the stats report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from data.corpus import RawPosting, _shingles, stratum_of
from data.schema import (
    FIELD_SPECS,
    JobPosting,
    build_messages,
    flatten,
    parse_prediction,
    to_target_json,
)

LOGGER = logging.getLogger("prepare")

SPLIT_SIZES = {"test": 500, "val": 500}  # train gets the remainder


# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------


def is_degenerate(obj: JobPosting) -> str | None:
    """Reject labels that are technically valid but carry no learning signal.

    Returns a rejection reason or None.

    WHY this matters more than it looks: an all-null label is *easy* to fit and
    numerous. Left in at their natural rate they push the model toward the
    constant-null solution, and they inflate per-field accuracy because null is
    the majority class. We cap rather than eliminate them -- some postings really
    do state almost nothing, and a model that cannot emit null is worse than one
    that over-emits it.
    """
    flat = flatten(obj)
    non_null = sum(1 for k, v in flat.items() if v not in (None, [], ""))
    if non_null <= 2:
        return "degenerate_all_null"
    if len(obj.job_title) < 3:
        return "title_too_short"
    if len(obj.required_skills) > 15 or len(obj.benefits) > 10:
        return "list_overflow"  # should be impossible post-validation; belt and braces
    return None


def filter_records(records: Iterable[dict[str, Any]], max_all_null_share: float = 0.10
                   ) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Apply quality gates and cap the share of near-empty labels."""
    kept: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    near_empty: list[dict[str, Any]] = []

    for rec in records:
        if rec.get("status") != "ok":
            rejected[f"teacher_{rec.get('status', 'unknown')}"] += 1
            continue
        obj, err = parse_prediction(rec["target_json"])
        if obj is None:
            rejected[f"reparse_{err}"] += 1
            continue
        reason = is_degenerate(obj)
        if reason == "degenerate_all_null":
            near_empty.append(rec)
            continue
        if reason:
            rejected[reason] += 1
            continue
        # Re-serialize through the schema so every target in the dataset is
        # byte-identical to what to_target_json produces. Guards against a
        # teacher label that validated but was not canonical (unsorted skills,
        # lowercase currency), which would otherwise teach the model two
        # different output conventions for the same input.
        rec["target_json"] = to_target_json(obj)
        kept.append(rec)

    cap = int(max_all_null_share * (len(kept) + len(near_empty)))
    for rec in near_empty[:cap]:
        obj, _ = parse_prediction(rec["target_json"])
        if obj is not None:
            rec["target_json"] = to_target_json(obj)
            kept.append(rec)
    rejected["degenerate_all_null_capped"] += max(0, len(near_empty) - cap)
    return kept, rejected


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def _bucket(key: str, mod: int = 10_000) -> int:
    """Stable hash bucket. md5 (not python hash) because PYTHONHASHSEED varies."""
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % mod


def split_records(
    records: list[dict[str, Any]],
    sizes: dict[str, int] = SPLIT_SIZES,
) -> dict[str, list[dict[str, Any]]]:
    """Grouped, stratified, hash-stable split, preferring gold labels for held-out.

    Four properties, in priority order:

    1. **Gold-labeled records fill val/test first.** Records carrying
       `_is_gold` were labeled with 3-sample self-consistency voting rather than
       a single greedy pass, so they are the best gold available. Spending them
       on the training split -- where SFT absorbs label noise fine -- and then
       measuring against single-pass labels would be exactly backwards.
    2. **Grouped by company_id** -- a company's postings never straddle a split.
       This is the strongest anti-leakage control available here, because the
       residual similarity after near-dup removal is company boilerplate.
    3. **Stratified by content stratum** (salary / deadline / plain) -- so the
       test split has the same hard-field positive rate as train and the metrics
       are comparable across splits.
    4. **Hash-stable** -- within each preference tier, assignment depends only on
       md5(company_id), so re-running after adding data never moves an example
       between splits.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        cid = str((rec.get("platform") or {}).get("company_id") or f"solo::{rec['posting_id']}")
        groups[cid].append(rec)

    # Sort key: gold groups first, then by hash. The hash term is deterministic
    # and uncorrelated with company size or corpus order, so no split gets
    # systematically bigger employers.
    def _rank(kv: tuple[str, list[dict[str, Any]]]) -> tuple[int, int]:
        cid, recs = kv
        all_gold = all(r.get("_is_gold") for r in recs)
        return (0 if all_gold else 1, _bucket(cid))

    ordered = sorted(groups.items(), key=_rank)

    assigned: dict[str, list[dict[str, Any]]] = {"test": [], "val": [], "train": []}
    # Fill test first, then val, then everything else to train. Filling the small
    # splits first guarantees they hit their target size exactly; the remainder
    # (train) absorbs the slack from indivisible group sizes.
    want = dict(sizes)
    for _cid, recs in ordered:
        for name in ("test", "val"):
            if want.get(name, 0) > 0 and len(recs) <= want[name]:
                assigned[name].extend(recs)
                want[name] -= len(recs)
                break
        else:
            assigned["train"].extend(recs)

    for name, recs in assigned.items():
        strata = Counter(stratum_of(RawPosting(r["posting_id"], r["source_text"])) for r in recs)
        LOGGER.info("split %-5s n=%-5d strata=%s", name, len(recs), dict(strata))
    return assigned


def cross_split_leakage(
    splits: dict[str, list[dict[str, Any]]], threshold: float = 0.6
) -> dict[str, Any]:
    """Verify no near-duplicate spans train and test. Reports, does not assume.

    Uses the same MinHash machinery as corpus dedup but at a *looser* threshold
    (0.6 vs 0.75): here we want to surface anything even loosely similar so it
    can be inspected, rather than silently drop it.
    """
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    for rec in splits["train"]:
        m = MinHash(num_perm=128)
        for sh in _shingles(rec["source_text"]):
            m.update(sh)
        lsh.insert(rec["posting_id"], m)

    hits: dict[str, list[str]] = {}
    for name in ("test", "val"):
        collisions = []
        for rec in splits[name]:
            m = MinHash(num_perm=128)
            for sh in _shingles(rec["source_text"]):
                m.update(sh)
            if lsh.query(m):
                collisions.append(rec["posting_id"])
        hits[name] = collisions

    train_cids = {str((r.get("platform") or {}).get("company_id")) for r in splits["train"]}
    company_overlap = {
        name: len(train_cids & {str((r.get("platform") or {}).get("company_id")) for r in splits[name]} - {"None"})
        for name in ("test", "val")
    }
    return {"near_dup_collisions": {k: len(v) for k, v in hits.items()},
            "collision_ids": {k: v[:10] for k, v in hits.items()},
            "shared_company_ids": company_overlap}


# ---------------------------------------------------------------------------
# SFT formatting
# ---------------------------------------------------------------------------


def to_sft_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Render one labeled example into TRL's conversational SFT format.

    We store `messages` rather than a pre-rendered string on purpose: the chat
    template lives in the tokenizer, so rendering here would hard-code Qwen's
    ChatML into the dataset and silently break if the base model is swapped. The
    training script applies `tokenizer.apply_chat_template` instead, and loss is
    masked to the assistant turn only (see train.py) so the model is never
    trained to reproduce the schema card.
    """
    msgs = build_messages(rec["source_text"])
    return {
        "posting_id": rec["posting_id"],
        "messages": [*msgs, {"role": "assistant", "content": rec["target_json"]}],
        "prompt": msgs[-1]["content"],
        "completion": rec["target_json"],
        "source_text": rec["source_text"],
        "target_json": rec["target_json"],
    }


def pick_fewshot_exemplars(train: list[dict[str, Any]], k: int = 3) -> list[dict[str, str]]:
    """Choose exemplars for the base-model few-shot arm. TRAIN ONLY, never test.

    Selection criteria, in order: short enough to fit three of them in context,
    and jointly covering the hard fields (one with a salary range, one with a
    deadline or experience requirement, one plain/mostly-null). The mostly-null
    exemplar is the important one -- it is what demonstrates the null-over-guess
    rule to an untrained model, and omitting it would make the few-shot baseline
    artificially weak and our lift artificially large.
    """
    def score(rec: dict[str, Any]) -> tuple[int, int]:
        return (len(rec["source_text"]), len(rec["target_json"]))

    pool = sorted((r for r in train if 800 <= len(r["source_text"]) <= 2500), key=score)
    want_salary, want_sparse, want_rich = None, None, None
    for rec in pool:
        flat = flatten(json.loads(rec["target_json"]))
        non_null = sum(1 for v in flat.values() if v not in (None, [], ""))
        if want_salary is None and flat.get("salary.min_amount") is not None:
            want_salary = rec
        elif want_sparse is None and non_null <= 5:
            want_sparse = rec
        elif want_rich is None and non_null >= 10:
            want_rich = rec
        if all((want_salary, want_sparse, want_rich)):
            break
    chosen = [r for r in (want_salary, want_rich, want_sparse) if r is not None][:k]
    chosen += [r for r in pool if r not in chosen][: k - len(chosen)]
    return [{"source_text": r["source_text"], "target_json": r["target_json"]} for r in chosen]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def dataset_stats(splits: dict[str, list[dict[str, Any]]]) -> str:
    """Per-field null rates and length distributions -- the dataset card body.

    Null rate per field is the single most useful number for reading the eval
    later: a field that is 92% null has a trivial majority-class baseline, and
    any accuracy figure for it must be read against that baseline rather than
    against 0.
    """
    lines = ["# Dataset statistics", ""]
    lines += ["| split | n | src chars p50 | src chars p95 | target chars p50 |", "|---|---:|---:|---:|---:|"]
    for name in ("train", "val", "test"):
        recs = splits[name]
        if not recs:
            continue
        src = sorted(len(r["source_text"]) for r in recs)
        tgt = sorted(len(r["target_json"]) for r in recs)
        q = lambda a, p: a[min(len(a) - 1, int(p * len(a)))]  # noqa: E731
        lines.append(f"| {name} | {len(recs)} | {q(src,.5)} | {q(src,.95)} | {q(tgt,.5)} |")

    lines += ["", "## Non-null rate per field", "",
              "Read every per-field metric in the results table against this column: "
              "a field that is 90% null has a 90% majority-class baseline.", "",
              "| field | tier | " + " | ".join(f"{s} non-null" for s in ("train", "val", "test")) + " |",
              "|---|---:|" + "---:|" * 3]
    for fname, spec in FIELD_SPECS.items():
        cells = []
        for name in ("train", "val", "test"):
            recs = splits[name]
            if not recs:
                cells.append("-")
                continue
            n = sum(1 for r in recs if flatten(json.loads(r["target_json"])).get(fname) not in (None, [], ""))
            cells.append(f"{n / len(recs):.0%}")
        lines.append(f"| `{fname}` | {spec['tier']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, action="append", default=None,
                    help="single-pass labeled records; repeatable (default: data/interim/labeled.jsonl)")
    ap.add_argument("--gold-from", type=Path, action="append", default=[],
                    help="records labeled with self-consistency voting; these are preferred for val/test")
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--stats-out", type=Path, default=Path("results/dataset_stats.md"))
    ap.add_argument("--test-size", type=int, default=500)
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--max-train", type=int, default=5000)
    ap.add_argument("--push-to-hub", default=None, help="e.g. username/structured-extract-jobs")
    ap.add_argument("--skip-leakage-check", action="store_true", help="the check is O(n) but not free on 5k docs")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from config import setup_run

    setup_run()

    inputs = args.inp or [Path("data/interim/labeled.jsonl")]
    missing = [p for p in [*inputs, *args.gold_from] if not p.exists()]
    if missing:
        LOGGER.error("missing input(s): %s -- run data.generate_synthetic first",
                     ", ".join(str(p) for p in missing))
        return 1

    # Gold records are loaded LAST and overwrite any single-pass label for the
    # same posting. The two phases draw from overlapping corpus slices, so a
    # posting can legitimately appear in both files; when it does, the voted
    # label is the better one and must win.
    by_id: dict[str, dict[str, Any]] = {}
    n_gold = 0
    for path in inputs:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                by_id[rec["posting_id"]] = {**rec, "_is_gold": False}
    for path in args.gold_from:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                by_id[rec["posting_id"]] = {**rec, "_is_gold": True}
                n_gold += 1
    raw = list(by_id.values())
    LOGGER.info("loaded %d unique labeled records (%d self-consistency-voted)", len(raw), n_gold)
    if args.gold_from and n_gold < args.test_size + args.val_size:
        LOGGER.warning(
            "only %d voted records for %d held-out slots -- val/test will be topped up with "
            "single-pass labels. Label more with --n-samples 3 for a cleaner held-out set.",
            n_gold, args.test_size + args.val_size)

    kept, rejected = filter_records(raw)
    LOGGER.info("quality filter: %d kept, rejections=%s", len(kept), dict(rejected))
    if len(kept) < args.test_size + args.val_size + 100:
        LOGGER.error("only %d usable examples; need at least %d. Label more postings.",
                     len(kept), args.test_size + args.val_size + 100)
        return 1

    splits = split_records(kept, {"test": args.test_size, "val": args.val_size})
    splits["train"] = splits["train"][: args.max_train]

    leakage: dict[str, Any] = {}
    if not args.skip_leakage_check:
        leakage = cross_split_leakage(splits)
        LOGGER.info("leakage check: %s", leakage["near_dup_collisions"])
        if leakage["near_dup_collisions"].get("test", 0) > 0:
            LOGGER.warning(
                "%d test postings resemble a training posting at Jaccard>=0.6. "
                "Inspect results/dataset_stats.md before trusting the test metrics.",
                leakage["near_dup_collisions"]["test"],
            )

    args.out.mkdir(parents=True, exist_ok=True)
    sft = {name: [to_sft_record(r) for r in recs] for name, recs in splits.items()}
    for name, recs in sft.items():
        path = args.out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        LOGGER.info("wrote %s (%d records)", path, len(recs))

    # The sacred-test-set receipt: an explicit id manifest, committed to git, so
    # any later claim that "the test set was never touched" is checkable.
    (args.out / "test_ids.txt").write_text(
        "\n".join(r["posting_id"] for r in sft["test"]) + "\n", encoding="utf-8"
    )

    exemplars = pick_fewshot_exemplars(splits["train"])
    (args.out / "fewshot_exemplars.json").write_text(
        json.dumps(exemplars, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("wrote %d few-shot exemplars (from TRAIN only)", len(exemplars))

    try:
        from datasets import Dataset, DatasetDict

        dd = DatasetDict({name: Dataset.from_list(recs) for name, recs in sft.items() if recs})
        dd.save_to_disk(str(args.out / "hf"))
        LOGGER.info("saved HF DatasetDict -> %s", args.out / "hf")
        if args.push_to_hub:
            dd.push_to_hub(args.push_to_hub, private=False)
            LOGGER.info("pushed -> https://huggingface.co/datasets/%s", args.push_to_hub)
    except ImportError:
        LOGGER.warning("`datasets` not installed; JSONL written but HF export skipped")

    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    body = dataset_stats(splits)
    gold_share = {name: sum(1 for r in recs if r.get("_is_gold")) for name, recs in splits.items()}
    LOGGER.info("self-consistency-voted labels per split: %s", gold_share)
    body += (
        "\n## Label provenance\n\n"
        "Held-out splits are filled from self-consistency-voted labels first (3 teacher "
        "samples, per-field majority vote, agreement gate on the hard fields). Training "
        "labels are single greedy passes -- SFT absorbs label noise, evaluation does not.\n\n"
        "| split | voted labels | single-pass | voted share |\n|---|---:|---:|---:|\n"
        + "".join(
            f"| {name} | {gold_share[name]} | {len(splits[name]) - gold_share[name]} | "
            f"{gold_share[name] / max(len(splits[name]), 1):.0%} |\n"
            for name in ("train", "val", "test")
        )
    )
    if leakage:
        body += (
            "\n## Leakage check\n\n"
            f"- near-duplicate collisions with train (Jaccard >= 0.6): "
            f"test={leakage['near_dup_collisions']['test']}, val={leakage['near_dup_collisions']['val']}\n"
            f"- company_ids shared with train: {leakage['shared_company_ids']}\n"
            f"- example colliding test ids: {leakage['collision_ids']['test']}\n"
        )
    body += "\n## Rejections during preparation\n\n| reason | count |\n|---|---:|\n"
    body += "".join(f"| `{k}` | {v} |\n" for k, v in rejected.most_common())
    args.stats_out.write_text(body, encoding="utf-8")
    LOGGER.info("wrote stats -> %s", args.stats_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
