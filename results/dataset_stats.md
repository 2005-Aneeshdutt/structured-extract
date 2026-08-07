# Dataset statistics

| split | n | src chars p50 | src chars p95 | target chars p50 |
|---|---:|---:|---:|---:|
| train | 2400 | 3566 | 7784 | 676 |
| val | 500 | 3540 | 7835 | 672 |
| test | 500 | 3573 | 7945 | 663 |

## Non-null rate per field

Read every per-field metric in the results table against this column: a field that is 90% null has a 90% majority-class baseline.

| field | tier | train non-null | val non-null | test non-null |
|---|---:|---:|---:|---:|
| `job_title` | 1 | 100% | 100% | 100% |
| `company_name` | 1 | 83% | 81% | 83% |
| `location.city` | 2 | 45% | 45% | 41% |
| `location.region` | 2 | 44% | 43% | 37% |
| `location.country` | 2 | 29% | 26% | 27% |
| `location.remote_policy` | 2 | 77% | 78% | 76% |
| `employment_type` | 3 | 80% | 78% | 79% |
| `seniority_level` | 3 | 90% | 88% | 89% |
| `education_requirement` | 3 | 74% | 71% | 72% |
| `required_skills` | 3 | 97% | 96% | 96% |
| `preferred_skills` | 3 | 77% | 76% | 75% |
| `benefits` | 3 | 58% | 59% | 50% |
| `salary.min_amount` | 4 | 32% | 34% | 30% |
| `salary.max_amount` | 4 | 26% | 30% | 26% |
| `salary.currency` | 4 | 27% | 30% | 25% |
| `salary.period` | 4 | 35% | 35% | 32% |
| `years_experience_min` | 4 | 62% | 65% | 63% |
| `application_deadline` | 4 | 5% | 5% | 4% |

## Label provenance

Held-out splits are filled from self-consistency-voted labels first (3 teacher samples, per-field majority vote, agreement gate on the hard fields). Training labels are single greedy passes -- SFT absorbs label noise, evaluation does not.

| split | voted labels | single-pass | voted share |
|---|---:|---:|---:|
| train | 217 | 2183 | 9% |
| val | 38 | 462 | 8% |
| test | 466 | 34 | 93% |

## Leakage check

- near-duplicate collisions with train (Jaccard >= 0.6): test=0, val=0
- company_ids shared with train: {'test': 0, 'val': 0}
- example colliding test ids: []

## Rejections during preparation

| reason | count |
|---|---:|
| `teacher_ungrounded` | 649 |
| `teacher_low_agreement` | 167 |
| `teacher_api_failed` | 5 |
| `title_too_short` | 3 |
| `teacher_parse_failed` | 1 |
| `degenerate_all_null_capped` | 0 |
