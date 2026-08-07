# Teacher label audit vs. platform metadata

Teacher: `google/gemini-2.5-flash-lite` via `openrouter`  |  labeled examples audited: 638

Agreement between teacher labels and LinkedIn's own structured form fields (independent of any LLM). Scored only where the teacher emitted a non-null value, so this measures label **precision**; a null where the platform has a value is usually correct, because the form field is often absent from the description text.

| field | n compared | agreement |
|---|---:|---:|
| `job_title` | 638 | 79.6% |
| `location.city` | 286 | 76.9% |
| `location.region` | 280 | 84.3% |
| `location.remote_policy` | 55 | 85.5% |
| `employment_type` | 529 | 91.1% |
| `seniority_level` | 431 | 88.9% |
| `salary.min_amount` | 146 | 94.5% |
| `salary.max_amount` | 142 | 97.2% |
| `salary.currency` | 177 | 100.0% |
| `salary.period` | 191 | 98.4% |

## Label funnel

| status | count | share |
|---|---:|---:|
| `ok` | 638 | 83.4% |
| `ungrounded` | 122 | 15.9% |
| `api_failed` | 5 | 0.7% |
