# Teacher label audit vs. platform metadata

Teacher: `google/gemini-2.5-flash-lite` via `openrouter`  |  labeled examples audited: 21

Agreement between teacher labels and LinkedIn's own structured form fields (independent of any LLM). Scored only where the teacher emitted a non-null value, so this measures label **precision**; a null where the platform has a value is usually correct, because the form field is often absent from the description text.

| field | n compared | agreement |
|---|---:|---:|
| `job_title` | 21 | 81.0% |
| `location.city` | 9 | 88.9% |
| `location.region` | 8 | 87.5% |
| `location.remote_policy` | 3 | 66.7% |
| `employment_type` | 16 | 100.0% |
| `seniority_level` | 16 | 100.0% |
| `salary.min_amount` | 4 | 100.0% |
| `salary.max_amount` | 4 | 100.0% |
| `salary.currency` | 5 | 100.0% |
| `salary.period` | 5 | 100.0% |

## Label funnel

| status | count | share |
|---|---:|---:|
| `ok` | 21 | 80.8% |
| `ungrounded` | 5 | 19.2% |
