# Teacher label audit vs. platform metadata

Teacher: `google/gemini-2.5-flash-lite` via `openrouter`  |  labeled examples audited: 723

Agreement between teacher labels and LinkedIn's own structured form fields (independent of any LLM). Scored only where the teacher emitted a non-null value, so this measures label **precision**; a null where the platform has a value is usually correct, because the form field is often absent from the description text.

| field | n compared | agreement |
|---|---:|---:|
| `job_title` | 723 | 79.1% |
| `location.city` | 318 | 80.5% |
| `location.region` | 298 | 86.2% |
| `location.remote_policy` | 71 | 93.0% |
| `employment_type` | 578 | 90.7% |
| `seniority_level` | 450 | 89.3% |
| `salary.min_amount` | 173 | 98.3% |
| `salary.max_amount` | 169 | 98.2% |
| `salary.currency` | 169 | 100.0% |
| `salary.period` | 212 | 99.5% |

## Label funnel

| status | count | share |
|---|---:|---:|
| `ok` | 723 | 72.3% |
| `low_agreement` | 167 | 16.7% |
| `ungrounded` | 109 | 10.9% |
| `parse_failed` | 1 | 0.1% |
