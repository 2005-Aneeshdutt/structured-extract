"""Gradio app construction tests.

`build_ui()` never touches llama-cpp (model loading is lazy in `_load`), so the
whole UI can be constructed in CI without a 1 GB GGUF. That is worth exercising:
the app was written against the Gradio 4.x API and a 6.x install removed
`Textbox(show_copy_button=)` and moved `theme=` off `Blocks` -- breakage that
would otherwise have surfaced as a crashed Space after deploy, not locally.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio", reason="gradio is not installed in the CI subset")

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import app as demo_app  # noqa: E402
from data.schema import MAX_SOURCE_CHARS, parse_prediction  # noqa: E402


def test_ui_builds_on_the_installed_gradio():
    assert demo_app.build_ui() is not None


class TestDemoExamples:
    def test_all_examples_are_usable(self):
        assert len(demo_app.EXAMPLES) >= 5, "the brief asks for 5-10 preloaded examples"
        for label, text in demo_app.EXAMPLES:
            assert label.strip(), "every example needs a label saying what it demonstrates"
            assert text.strip(), f"{label}: empty input"

    def test_examples_fit_the_training_context(self):
        # An example longer than the truncation budget would demo the model on
        # input it was never trained to see, which is a misleading demo.
        for label, text in demo_app.EXAMPLES:
            assert len(text) <= MAX_SOURCE_CHARS, f"{label}: {len(text)} chars exceeds the budget"

    def test_examples_cover_the_hard_cases(self):
        """The examples must actually exercise the failure modes they claim to."""
        blob = " ".join(t for _lab, t in demo_app.EXAMPLES).lower()
        assert "per hour" in blob            # hourly normalization
        assert "k–" in blob or "k-" in blob  # k-suffix parsing
        assert "£" in blob or "cad" in blob  # non-USD currency
        assert "preferred" in blob           # required-vs-preferred reasoning
        # At least one example must state no salary at all -- that is the case
        # where the base model hallucinates, i.e. the point of the demo.
        assert any("salary" not in t.lower() and "$" not in t and "pay" not in t.lower()
                   for _lab, t in demo_app.EXAMPLES)


class TestGuards:
    def test_empty_input_does_not_load_a_model(self):
        _out, status, _raw = demo_app.extract("   ", "Fine-tuned (LoRA r=16)")
        assert "Paste a job posting" in status
        assert not demo_app._MODELS, "no model should be loaded for empty input"

    def test_unconfigured_repo_fails_with_a_readable_message(self, monkeypatch):
        """Only meaningful on the GGUF path -- FT_REPO names a GGUF repo.

        The transformers fallback reads ADAPTER_REPO instead, so firing this
        guard there would block a correctly-configured machine that simply has
        no llama.cpp. Forced on so the message is tested wherever the suite runs.
        """
        monkeypatch.setattr(demo_app, "_have_llama_cpp", lambda: True)
        monkeypatch.setattr(demo_app, "FINETUNED_REPO", "SET_HF_USER/whatever")
        _out, status, _raw = demo_app.extract("Engineer at Acme", "Fine-tuned (LoRA r=16)")
        assert "Configuration needed" in status
        assert "FT_REPO" in status
        assert not demo_app._MODELS

    def test_fallback_path_is_not_blocked_by_the_gguf_guard(self, monkeypatch):
        """Without llama.cpp the GGUF repo id is irrelevant and must not gate."""
        monkeypatch.setattr(demo_app, "_have_llama_cpp", lambda: False)
        monkeypatch.setattr(demo_app, "FINETUNED_REPO", "SET_HF_USER/whatever")
        called = {}
        monkeypatch.setattr(demo_app, "_complete",
                            lambda which, text: called.setdefault("which", which) or '{"job_title":"x"}')
        _out, status, _raw = demo_app.extract("Engineer at Acme", "Fine-tuned (LoRA r=16)")
        assert "Configuration needed" not in status
        assert called["which"] == "finetuned"

    def test_runtime_label_names_what_served_the_request(self, monkeypatch):
        """The two runtimes are not numerically identical, so the UI says which."""
        monkeypatch.setattr(demo_app, "_have_llama_cpp", lambda: True)
        assert "GGUF" in demo_app.runtime_label()
        monkeypatch.setattr(demo_app, "_have_llama_cpp", lambda: False)
        assert "adapter" in demo_app.runtime_label()


def test_app_uses_the_shared_schema_not_a_copy():
    """The demo must import the same prompt the model was trained on.

    A vendored copy in the Space is how a demo silently drifts from the evaluated
    system -- the page would show numbers the README cannot reproduce.
    """
    from data.schema import SYSTEM_PROMPT, build_messages

    assert demo_app.SYSTEM_PROMPT is SYSTEM_PROMPT
    assert demo_app.build_messages is build_messages
    assert demo_app.parse_prediction is parse_prediction
