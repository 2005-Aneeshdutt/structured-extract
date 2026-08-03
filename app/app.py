"""Gradio demo: paste a job posting, get validated JSON.

    python app/app.py                                  # downloads GGUFs from the Hub
    python app/app.py --local models/                  # use local GGUF files
    python app/app.py --share                          # public tunnel

Deployed to HuggingFace Spaces on the free CPU tier. Both models are ~1 GB GGUF
Q4_K_M, which is why a 1.5B model can serve interactive traffic on 2 vCPUs with
no GPU at all -- the entire point of the quantization step.

The demo is built to make the *comparison* legible, not just to show off the
fine-tune: the model switch keeps the input, so a visitor can run the same
posting through both and watch the base model invent a salary that the posting
never mentions. The preloaded examples are chosen to trigger exactly that.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import gradio as gr

# The Space has app.py at its root, while the repo has it under app/. Adding the
# parent directory covers both layouts, so the SAME file runs in both places --
# and, critically, imports the same data/schema.py the model was trained against.
# A vendored copy of the schema here is how a demo silently drifts from the
# evaluated system.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if (_REPO_ROOT / "data" / "schema.py").exists():
    sys.path.insert(0, str(_REPO_ROOT))

from data.schema import (
    SYSTEM_PROMPT,
    build_messages,
    parse_prediction,
    truncate_source,
    ungrounded_fields,
)

# Hub locations. Set FT_REPO in the Space's variables (or your .env) -- the
# default is an obvious placeholder that fails loudly rather than silently
# pulling someone else's weights.
FINETUNED_REPO = os.environ.get("FT_REPO", "SET_HF_USER/qwen2.5-1.5b-jobs-extract-GGUF")
FINETUNED_FILE = os.environ.get("FT_FILE", "qwen2.5-1.5b-r16-a32-Q4_K_M.gguf")
BASE_REPO = os.environ.get("BASE_REPO", "Qwen/Qwen2.5-1.5B-Instruct-GGUF")
BASE_FILE = os.environ.get("BASE_FILE", "qwen2.5-1.5b-instruct-q4_k_m.gguf")

MAX_NEW_TOKENS = 400
N_CTX = 4096

_MODELS: dict[str, Any] = {}
_LOCAL_DIR: Path | None = None


def _load(which: str):
    """Lazily load and cache a llama.cpp model.

    Lazy because a Space that loads two 1 GB models at import time takes ~40s to
    boot and gets killed by the health check. Loading on first use means the page
    is interactive immediately and only the model the visitor actually picks is
    ever paid for.
    """
    if which in _MODELS:
        return _MODELS[which]
    from llama_cpp import Llama

    repo, fname = (FINETUNED_REPO, FINETUNED_FILE) if which == "finetuned" else (BASE_REPO, BASE_FILE)
    if _LOCAL_DIR is not None:
        path = str(_LOCAL_DIR / fname)
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=repo, filename=fname)

    _MODELS[which] = Llama(
        model_path=path,
        n_ctx=N_CTX,
        n_threads=int(os.environ.get("N_THREADS", "2")),  # free Spaces tier = 2 vCPU
        n_gpu_layers=0,
        verbose=False,
        seed=0,
    )
    return _MODELS[which]


def extract(posting_text: str, model_choice: str) -> tuple[str, str, str]:
    """Returns (json_output, status_markdown, raw_output)."""
    if not posting_text or not posting_text.strip():
        return "{}", "Paste a job posting on the left, then press **Extract**.", ""

    which = "finetuned" if model_choice.startswith("Fine") else "base"
    if which == "finetuned" and FINETUNED_REPO.startswith("SET_HF_USER"):
        return ("{}", "**Configuration needed.** Set the `FT_REPO` environment variable to the "
                "HuggingFace repo holding your GGUF (e.g. `yourname/qwen2.5-1.5b-jobs-extract-GGUF`), "
                "or run with `--local models/`.", "")
    llm = _load(which)

    t0 = time.perf_counter()
    resp = llm.create_chat_completion(
        messages=build_messages(posting_text),
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,  # greedy, so the demo is reproducible for a visitor
    )
    elapsed = time.perf_counter() - t0
    raw = resp["choices"][0]["message"]["content"] or ""

    obj, err = parse_prediction(raw)
    if obj is None:
        # Failure is shown, not hidden. Watching the base model emit a fenced,
        # half-finished object is the most persuasive part of the demo.
        return (
            raw.strip() or "(empty output)",
            f"❌ **Did not produce valid JSON** — `{err}`\n\n"
            f"⏱ {elapsed:.2f}s · model: `{model_choice}`",
            raw,
        )

    pretty = json.dumps(obj.model_dump(mode="json"), indent=2, ensure_ascii=False)
    ungrounded = ungrounded_fields(obj, posting_text)
    n_filled = sum(1 for v in obj.model_dump().values() if v not in (None, [], {}))

    status = [f"✅ **Valid JSON**, schema-conformant · {n_filled}/12 top-level fields populated",
              f"⏱ {elapsed:.2f}s · model: `{model_choice}` · Q4_K_M GGUF on CPU"]
    if ungrounded:
        # The same grounding check used to compute the hallucination metric in
        # eval/. Surfacing it live is the point: it turns an abstract number in
        # the README into something a visitor can watch happen.
        status.append(
            "⚠️ **Unsupported values** (no matching span in the source text): "
            + ", ".join(f"`{f}`" for f in ungrounded)
        )
    if len(posting_text) > len(truncate_source(posting_text)):
        status.append(f"ℹ️ Input truncated to {len(truncate_source(posting_text)):,} chars "
                      "(the model's training context).")
    return pretty, "\n\n".join(status), raw


# ---------------------------------------------------------------------------
# Demo inputs
#
# Each one targets a specific, documented failure mode of the base model. They
# are written for the demo rather than drawn from the corpus, so nothing here
# leaks a test example into a public page.
# ---------------------------------------------------------------------------

EXAMPLES: list[tuple[str, str]] = [
    (
        "No salary stated → base model invents one",
        """Senior Backend Engineer
Northwind Analytics — Austin, TX (Hybrid, 3 days onsite)

We're looking for a Senior Backend Engineer to join our platform team. You'll own
services that process billions of events per day.

What you'll need:
- 5+ years building production backend systems
- Strong Python and PostgreSQL
- Experience with Kubernetes and CI/CD pipelines
- Bachelor's degree in Computer Science or equivalent experience

Nice to have:
- Go, Kafka
- Prior fintech experience

We offer health insurance, 401k matching, and unlimited PTO.""",
    ),
    (
        "Hourly rate + range → numeric normalization",
        """Warehouse Associate (Part-Time) - Night Shift
Cascade Logistics | Kent, WA | On-site

Pay: $22.50 - $26.75 per hour, depending on experience. Shift differential of
$1.50/hr for overnight shifts.

Requirements: Ability to lift 50 lbs, forklift certification preferred.
High school diploma or GED required. No prior warehouse experience necessary.

Benefits: medical, dental, paid time off, employee discount.""",
    ),
    (
        "Salary written as '$120k–$150k' → k-suffix parsing",
        """Staff Machine Learning Engineer (Remote - US)
Helio AI

Compensation: $120k–$150k base, plus equity.

About the role: You will lead model training infrastructure. Requires 8 years of
industry experience, deep PyTorch expertise, and a track record shipping ML
systems at scale. PhD preferred but not required.

Application deadline: 2025-03-14.""",
    ),
    (
        "Required vs preferred → section-boundary reasoning",
        """Data Analyst
BrightPath Health, Chicago IL

MUST HAVE: SQL, Excel, Tableau, 2 years of analytics experience.
PREFERRED: Python, dbt, healthcare domain knowledge, Snowflake.

This is a full-time, onsite position. Master's degree required.""",
    ),
    (
        "Sparse stub → should return mostly nulls",
        """Line Cook needed. Apply in person.""",
    ),
    (
        "ALL CAPS + broken formatting → robustness",
        """CUSTOMER SUCCESS MANAGER!!! ***URGENT HIRE***
LOCATION: REMOTE (ANYWHERE IN CANADA)EMPLOYMENT TYPE: FULL TIME PERMANENT
WE NEED SOMEONE WITH 3-5 YEARS OF SAAS CUSTOMER SUCCESS EXPERIENCE.
SKILLS: SALESFORCE, ZENDESK, EXCELLENT COMMUNICATION
COMPENSATION: CAD 75,000 - 90,000 ANNUALLY + BONUS
PERKS: REMOTE WORK, HEALTH BENEFITS, PROFESSIONAL DEVELOPMENT BUDGET""",
    ),
    (
        "Non-USD currency → ISO-4217 mapping",
        """Frontend Developer (React)
Lumen Studios — London, United Kingdom — Hybrid

£45,000 – £58,000 per annum.

You will build customer-facing interfaces in React and TypeScript. We ask for at
least 3 years of commercial frontend experience. Familiarity with Next.js and
GraphQL is a bonus. Degree not required — we care about your portfolio.

Benefits: 28 days holiday, pension contribution, cycle-to-work scheme.""",
    ),
    (
        "Salary in boilerplate tail → distractor text",
        """Registered Nurse - Med/Surg Unit
Riverside Community Hospital, Riverside, CA
Full-time, night shift, 12-hour rotations.

Qualifications: Active RN license, BSN preferred, minimum 1 year acute care
experience. BLS and ACLS certification required.

Riverside Community Hospital is an equal opportunity employer. All qualified
applicants will receive consideration for employment without regard to race,
color, religion, sex, national origin, disability or protected veteran status.
We participate in E-Verify. This position is covered by a collective bargaining
agreement. The pay range for this position is $42.00 to $61.50 per hour. Actual
pay will be determined by experience and internal equity.""",
    ),
]


def build_ui(default_model: str = "Fine-tuned (LoRA r=16)") -> gr.Blocks:
    # Deliberately minimal Gradio API surface so the same file runs on 4.x, 5.x
    # and 6.x. Two things were removed after testing against 6.22:
    #   * `theme=` on Blocks -- moved to launch() in Gradio 6, and there is no
    #     spelling that works on both. The default theme is fine.
    #   * `show_copy_button=` on Textbox -- removed in Gradio 6.
    # Both were cosmetic; neither is worth a version pin that could leave the
    # Space and the local install on different majors.
    with gr.Blocks(title="Structured Extraction — Qwen2.5-1.5B LoRA") as demo:
        gr.Markdown(
            "# Structured JSON extraction from job postings\n"
            "A **1.5B** parameter model, LoRA fine-tuned and quantized to a **1 GB GGUF**, "
            "extracting a 12-field schema on CPU. Switch models to see what the fine-tune "
            "actually bought — the base model tends to wrap output in markdown, drop required "
            "keys, and invent salaries that the posting never states.\n\n"
            "[Code & full evaluation](https://github.com/2005-Aneeshdutt/structured-extract) · "
            "the metrics in the README are computed by the same functions this page calls."
        )
        with gr.Row():
            with gr.Column(scale=1):
                model_choice = gr.Radio(
                    ["Fine-tuned (LoRA r=16)", "Base Qwen2.5-1.5B-Instruct"],
                    value=default_model,
                    label="Model",
                    info="Both are Q4_K_M GGUF, same prompt, greedy decoding.",
                )
                posting = gr.Textbox(
                    label="Raw job posting",
                    placeholder="Paste any unstructured job posting here…",
                    lines=22,
                    max_lines=40,
                )
                with gr.Row():
                    run_btn = gr.Button("Extract", variant="primary", scale=2)
                    clear_btn = gr.ClearButton(value="Clear", scale=1)
            with gr.Column(scale=1):
                status = gr.Markdown("Paste a posting and press **Extract**.")
                output = gr.Code(label="Extracted JSON", language="json", lines=24)
                with gr.Accordion("Raw model output (before parsing)", open=False):
                    raw_out = gr.Textbox(label="", lines=8)

        gr.Markdown("### Try one of these — each targets a specific base-model failure mode")
        gr.Examples(
            examples=[[text, label] for label, text in EXAMPLES],
            inputs=[posting, gr.Textbox(visible=False)],
            label="",
            examples_per_page=8,
        )

        with gr.Accordion("The prompt (identical for both models, and for every number in the README)", open=False):
            gr.Code(SYSTEM_PROMPT, label="System prompt", language=None)

        run_btn.click(extract, inputs=[posting, model_choice], outputs=[output, status, raw_out])
        clear_btn.add([posting, output, raw_out])
    return demo


def main() -> None:
    global _LOCAL_DIR

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local", type=Path, default=None, help="directory holding local .gguf files")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    _LOCAL_DIR = args.local
    build_ui().launch(share=args.share, server_port=args.port, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
