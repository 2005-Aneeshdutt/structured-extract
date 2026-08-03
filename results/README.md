# results/

Everything in this directory is **generated**, and everything except
`raw_predictions/` is **committed** — the tables and charts are the point of the
project, so they belong in git where a reader can see them without running
anything.

| File | Produced by | Contents |
|---|---|---|
| `dataset_stats.md` | `make prepare` | per-split sizes, per-field non-null rates, **leakage check** |
| `label_audit.md` | `make label` | teacher labels vs LinkedIn's non-LLM metadata + the label funnel |
| `comparison_table.md` | `make compare` | 4-model table, bootstrap CIs, gap-closed, paired significance, per-field F1 |
| `comparison.json` | `make compare` | the same numbers, machine-readable |
| `ablation_table.md` | `make report` | rank 8 / 16 / 32 on the **validation** split |
| `failure_analysis.md` | `make report` | 7-category taxonomy + 10 worked examples |
| `quantization_check.md` | `make verify` | fp16 + adapter vs GGUF Q4_K_M, with per-field disagreement |
| `charts/*.png` | `make report` | light **and** dark variants of every figure |
| `raw_predictions/*.json` | `make eval-all` | every completion, gitignored (large), kept locally as evidence |
| `corpus_facts.json` | measured once over the full corpus | the numbers every docstring and README claim cites — length percentiles, dedup rate, stratum availability |

`corpus_facts.json` is committed because several design decisions in the code
(the 6,000-char truncation budget, `STRATUM_TARGETS`, the low-support caveat on
`application_deadline`) are justified by those specific measurements. Regenerate
with:

```python
from data.corpus import load_corpus, describe_lengths
print(describe_lengths(load_corpus(25000)))
```

## Why raw predictions are kept

Scoring is separated from generation on purpose. Generation is the expensive
step; scoring is cheap and pure. Keeping the completions means a metric bug can
be fixed and every table rebuilt with `make compare report` — no GPU time, no
further Gemini quota. It also means the numbers are auditable: anyone can
re-score the saved outputs and get the same table.

## A note on reading these files

Two numbers deserve suspicion by default, and both are labeled in place:

- The **Gemini column** is biased upward, because Gemini also produced the gold
  labels. Gap-closed figures are therefore conservative.
- Any per-field row whose **support** is small (notably `application_deadline`)
  has a wide interval. Read it alongside the non-null rates in
  `dataset_stats.md` rather than on its own.
