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


# ---------------------------------------------------------------------------
# Presentation
#
# Delivered as a CSS string rather than a gr.themes object on purpose: `theme=`
# moved between Blocks and launch() across Gradio 4/5/6 with no spelling that
# works on all three, whereas `css=` has been stable throughout. Every rule is
# anchored on an elem_id/elem_classes WE set, or on `.gradio-container`, so a
# Gradio release that renames its internal DOM classes degrades the styling
# rather than breaking the page.
#
# The design commits to dark. It defines its own palette instead of inheriting
# Gradio's light/dark switch, because a glass-and-glow treatment that has to
# read on both grounds ends up committing to neither.
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root, .dark {
  --se-bg:        #05070d;
  --se-panel:     rgba(19, 25, 41, 0.66);
  --se-border:    rgba(120, 190, 255, 0.16);
  --se-border-hi: rgba(120, 220, 255, 0.42);
  --se-text:      #dbe6f5;
  --se-muted:     #8194b3;
  --se-cyan:      #35e6ff;
  --se-violet:    #9b7bff;
  --se-lime:      #6ef2a8;
  --se-amber:     #ffc861;
  --se-rose:      #ff7a92;
  --se-sans: 'Space Grotesk', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif;
  --se-mono: 'JetBrains Mono', ui-monospace, 'Cascadia Code', Consolas, monospace;
}

/* ---- ground ---------------------------------------------------------- */
.gradio-container, .gradio-container .main, body {
  background: var(--se-bg) !important;
  color: var(--se-text) !important;
  font-family: var(--se-sans) !important;
}

/* Aurora + grid. Fixed and pointer-events:none so it never intercepts
   clicks or scrolls with the content. */
.gradio-container::before {
  content: ""; position: fixed; inset: -30%; z-index: 0; pointer-events: none;
  background:
    radial-gradient(38rem 30rem at 12% 8%,  rgba(53,230,255,0.16), transparent 62%),
    radial-gradient(34rem 28rem at 88% 4%,  rgba(155,123,255,0.15), transparent 60%),
    radial-gradient(40rem 34rem at 70% 96%, rgba(110,242,168,0.09), transparent 64%);
  filter: blur(6px);
  animation: se-drift 26s ease-in-out infinite alternate;
}
.gradio-container::after {
  content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image:
    linear-gradient(rgba(120,190,255,0.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(120,190,255,0.045) 1px, transparent 1px);
  background-size: 46px 46px;
  mask-image: radial-gradient(circle at 50% 22%, #000 0%, transparent 78%);
  -webkit-mask-image: radial-gradient(circle at 50% 22%, #000 0%, transparent 78%);
}
@keyframes se-drift {
  from { transform: translate3d(-1.5%, -1%, 0) scale(1.02); }
  to   { transform: translate3d(1.5%, 1.5%, 0) scale(1.08); }
}
/* Anything above the backdrop needs its own stacking context. */
.gradio-container > * { position: relative; z-index: 1; }

/* Respect a stated preference for less motion. */
@media (prefers-reduced-motion: reduce) {
  .gradio-container::before { animation: none; }
}

/* ---- masthead -------------------------------------------------------- */
#se-hero { text-align: center; padding: 2.4rem 1rem 0.4rem; }
#se-hero h1 {
  font-size: clamp(1.9rem, 4.4vw, 3.1rem); font-weight: 700; letter-spacing: -0.02em;
  line-height: 1.08; margin: 0 0 0.5rem;
  background: linear-gradient(96deg, var(--se-cyan) 4%, #b8d8ff 42%, var(--se-violet) 96%);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; color: transparent;
}
#se-hero .se-kicker {
  font-family: var(--se-mono); font-size: 0.7rem; letter-spacing: 0.34em;
  text-transform: uppercase; color: var(--se-cyan); opacity: 0.85; margin-bottom: 0.9rem;
}
#se-hero p { color: var(--se-muted); max-width: 64ch; margin: 0 auto; line-height: 1.65; }
#se-hero a { color: var(--se-cyan); text-decoration: none; border-bottom: 1px solid rgba(53,230,255,0.3); }
#se-hero a:hover { border-bottom-color: var(--se-cyan); }

.se-specs { display: flex; flex-wrap: wrap; gap: 0.5rem; justify-content: center; margin: 1.4rem 0 0.4rem; }
.se-spec {
  font-family: var(--se-mono); font-size: 0.74rem; padding: 0.4rem 0.85rem;
  border: 1px solid var(--se-border); border-radius: 999px;
  background: rgba(255,255,255,0.03); color: var(--se-muted);
  backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
}
.se-spec b { color: var(--se-text); font-weight: 500; }

/* ---- glass panels ---------------------------------------------------- */
.se-card {
  background: var(--se-panel) !important;
  border: 1px solid var(--se-border) !important;
  border-radius: 16px !important;
  backdrop-filter: blur(16px) saturate(140%); -webkit-backdrop-filter: blur(16px) saturate(140%);
  box-shadow: 0 1px 0 rgba(255,255,255,0.05) inset, 0 18px 46px rgba(0,0,0,0.5);
  padding: 1rem !important;
}
.se-card:focus-within { border-color: var(--se-border-hi) !important; }

/* Gradio's own block chrome, flattened so our card is the only frame. */
.se-card .block, .se-card .form, .se-card .wrap {
  background: transparent !important; border: none !important; box-shadow: none !important;
}
.gradio-container label span, .gradio-container .label-wrap span { color: var(--se-muted) !important; }

/* ---- inputs ---------------------------------------------------------- */
.gradio-container textarea, .gradio-container input[type=text] {
  background: rgba(6,10,20,0.72) !important;
  border: 1px solid var(--se-border) !important;
  border-radius: 12px !important;
  color: var(--se-text) !important;
  font-family: var(--se-mono) !important; font-size: 0.84rem !important; line-height: 1.6 !important;
  transition: border-color .18s ease, box-shadow .18s ease;
}
.gradio-container textarea:focus, .gradio-container input[type=text]:focus {
  border-color: var(--se-cyan) !important;
  box-shadow: 0 0 0 3px rgba(53,230,255,0.13) !important; outline: none !important;
}

/* ---- primary action -------------------------------------------------- */
#se-run {
  background: linear-gradient(96deg, var(--se-cyan), var(--se-violet)) !important;
  border: none !important; border-radius: 12px !important;
  color: #04121b !important; font-weight: 700 !important; letter-spacing: 0.03em;
  font-family: var(--se-sans) !important; font-size: 0.95rem !important;
  padding: 0.8rem 1.2rem !important;
  box-shadow: 0 8px 26px rgba(53,230,255,0.26);
  transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
}
#se-run:hover { transform: translateY(-1px); box-shadow: 0 12px 34px rgba(53,230,255,0.4); filter: brightness(1.06); }
#se-run:active { transform: translateY(0); }

#se-clear {
  background: rgba(255,255,255,0.045) !important;
  border: 1px solid var(--se-border) !important; border-radius: 12px !important;
  color: var(--se-muted) !important;
}
#se-clear:hover { border-color: var(--se-border-hi) !important; color: var(--se-text) !important; }

/* ---- output ---------------------------------------------------------- */
#se-json, #se-json * { font-family: var(--se-mono) !important; font-size: 0.82rem !important; }
#se-json .cm-editor, #se-json .cm-scroller, #se-json pre, #se-json code {
  background: rgba(4,8,16,0.85) !important;
}
#se-json { border: 1px solid var(--se-border) !important; border-radius: 12px !important; overflow: hidden; }

#se-status { min-height: 3.2rem; font-size: 0.9rem; line-height: 1.7; }
#se-status code {
  font-family: var(--se-mono); background: rgba(53,230,255,0.09);
  border: 1px solid rgba(53,230,255,0.18); border-radius: 6px;
  padding: 0.1rem 0.38rem; color: var(--se-cyan);
}
#se-status strong { color: var(--se-text); }

/* ---- radio / accordion / examples ------------------------------------ */
.gradio-container input[type=radio] { accent-color: var(--se-cyan); }
.gradio-container .accordion, .gradio-container details {
  background: rgba(255,255,255,0.022) !important;
  border: 1px solid var(--se-border) !important; border-radius: 12px !important;
}
.gradio-container table, .gradio-container .table-wrap {
  background: transparent !important; border-color: var(--se-border) !important;
}
.gradio-container tbody tr { transition: background .14s ease; }
.gradio-container tbody tr:hover { background: rgba(53,230,255,0.06) !important; }

.se-section-title {
  font-family: var(--se-mono); font-size: 0.72rem; letter-spacing: 0.24em;
  text-transform: uppercase; color: var(--se-cyan); opacity: 0.8;
  margin: 1.8rem 0 0.5rem;
}

footer, .gradio-container footer { display: none !important; }
"""

HERO = """
<div id="se-hero">
  <div class="se-kicker">Qwen2.5 · LoRA r=16 · Q4_K_M</div>
  <h1>Structured extraction from messy job postings</h1>
  <p>
    A <b>1.5-billion-parameter</b> model, LoRA fine-tuned and quantized to a
    <b>1&nbsp;GB</b> file, pulling a 12-field schema out of free text — running on
    CPU, with no API call. Switch models to see what the fine-tune bought: the
    base model wraps output in markdown, drops required keys, and invents
    salaries the posting never mentions.
  </p>
  <div class="se-specs">
    <span class="se-spec"><b>1.5B</b> params</span>
    <span class="se-spec"><b>18</b> scored fields</span>
    <span class="se-spec"><b>Q4_K_M</b> GGUF</span>
    <span class="se-spec"><b>CPU</b> inference</span>
    <span class="se-spec"><b>0</b> API calls</span>
  </div>
  <p style="margin-top:1rem;font-size:0.86rem">
    <a href="https://github.com/2005-Aneeshdutt/structured-extract">Code &amp; full evaluation</a>
    — every metric in the README is computed by the same functions this page calls.
  </p>
</div>
"""


def build_ui(default_model: str = "Fine-tuned (LoRA r=16)") -> gr.Blocks:
    # Deliberately minimal Gradio API surface so the same file runs on 4.x, 5.x
    # and 6.x. Two things were removed after testing against 6.22:
    #   * `theme=` on Blocks -- moved to launch() in Gradio 6, and there is no
    #     spelling that works on both. The default theme is fine.
    #   * `show_copy_button=` on Textbox -- removed in Gradio 6.
    # Both were cosmetic; neither is worth a version pin that could leave the
    # Space and the local install on different majors.
    with gr.Blocks(title="Structured Extraction — Qwen2.5-1.5B LoRA", css=CSS) as demo:
        gr.HTML(HERO)
        with gr.Row():
            with gr.Column(scale=1, elem_classes="se-card"):
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
                    run_btn = gr.Button("Extract →", variant="primary", scale=2, elem_id="se-run")
                    clear_btn = gr.ClearButton(value="Clear", scale=1, elem_id="se-clear")
            with gr.Column(scale=1, elem_classes="se-card"):
                status = gr.Markdown("Paste a posting and press **Extract**.", elem_id="se-status")
                output = gr.Code(label="Extracted JSON", language="json", lines=24, elem_id="se-json")
                with gr.Accordion("Raw model output (before parsing)", open=False):
                    raw_out = gr.Textbox(label="", lines=8)

        gr.HTML('<div class="se-section-title">Try one — each targets a documented base-model failure</div>')
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
