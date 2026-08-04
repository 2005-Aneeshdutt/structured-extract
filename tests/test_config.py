"""Credential loading tests.

The bug these guard against already happened once: `python-dotenv` was a declared
dependency, but `load_dotenv()` was never called. Pasting a key into `.env` had
no effect, and the resulting error ("GOOGLE_API_KEY not set") gave no hint that
the file you just edited is never read.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import config


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """load_project_env caches after the first call; reset between tests."""
    monkeypatch.setattr(config, "_loaded", False)
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "HF_TOKEN",
                "OPENROUTER_API_KEY", "REQUESTS_PER_DAY", "REQUESTS_PER_MINUTE"):
        monkeypatch.delenv(var, raising=False)


def _write_env(tmp_path: Path, body: str) -> Path:
    p = tmp_path / ".env"
    p.write_text(body, encoding="utf-8")
    return p


class TestLoading:
    def test_reads_key_from_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "ENV_PATH", _write_env(tmp_path, "GOOGLE_API_KEY=abc123\n"))
        assert config.get_api_key("GOOGLE_API_KEY") == "abc123"

    def test_missing_file_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "ENV_PATH", tmp_path / "nonexistent")
        assert config.load_project_env() is False
        assert config.get_api_key("GOOGLE_API_KEY") is None

    def test_real_env_var_wins_over_file(self, tmp_path, monkeypatch):
        """CI and Kaggle inject secrets as env vars; a stale .env must not win."""
        monkeypatch.setattr(config, "ENV_PATH", _write_env(tmp_path, "GOOGLE_API_KEY=from_file\n"))
        monkeypatch.setenv("GOOGLE_API_KEY", "from_environment")
        assert config.get_api_key("GOOGLE_API_KEY") == "from_environment"

    def test_ignores_comments_blanks_and_quotes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "ENV_PATH", _write_env(
            tmp_path, '# a comment\n\nGOOGLE_API_KEY="quoted-key"\n  \nHF_TOKEN=tok\n'))
        assert config.get_api_key("GOOGLE_API_KEY") == "quoted-key"
        assert config.get_hf_token() == "tok"

    def test_falls_back_to_the_first_name_that_has_a_value(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "ENV_PATH", _write_env(tmp_path, "GEMINI_API_KEY=legacy\n"))
        assert config.get_api_key("GOOGLE_API_KEY", "GEMINI_API_KEY") == "legacy"


class TestIntSettings:
    """`REQUESTS_PER_DAY`/`REQUESTS_PER_MINUTE` shipped in .env for a while with
    no reader. These tests exist so the setting stays wired to the budget."""

    def test_reads_an_integer_from_the_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "ENV_PATH", _write_env(tmp_path, "REQUESTS_PER_DAY=8000\n"))
        assert config.get_int("REQUESTS_PER_DAY") == 8000

    def test_blank_means_use_the_default(self, tmp_path, monkeypatch):
        """Blank must be None, not 0 -- a 0/day budget silently labels nothing."""
        monkeypatch.setattr(config, "ENV_PATH", _write_env(tmp_path, "REQUESTS_PER_DAY=\n"))
        assert config.get_int("REQUESTS_PER_DAY") is None

    def test_unset_is_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "ENV_PATH", tmp_path / "nonexistent")
        assert config.get_int("REQUESTS_PER_DAY") is None

    def test_garbage_raises_rather_than_falling_back(self, tmp_path, monkeypatch):
        """A typo'd budget must not silently revert to the built-in default."""
        monkeypatch.setattr(config, "ENV_PATH", _write_env(tmp_path, "REQUESTS_PER_DAY=1,500\n"))
        with pytest.raises(SystemExit, match="must be an integer"):
            config.get_int("REQUESTS_PER_DAY")


class TestOpenRouterKey:
    def test_reads_the_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "ENV_PATH",
                            _write_env(tmp_path, "OPENROUTER_API_KEY=sk-or-v1-abc\n"))
        assert config.require_openrouter_key() == "sk-or-v1-abc"

    def test_missing_key_names_the_file_and_the_quota_doc(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "ENV_PATH", _write_env(tmp_path, "OPENROUTER_API_KEY=\n"))
        with pytest.raises(SystemExit) as exc:
            config.require_openrouter_key()
        msg = str(exc.value)
        assert str(tmp_path / ".env") in msg
        assert "limits" in msg, "the credit-gated daily cap is the thing users trip on"

    def test_gemini_key_does_not_satisfy_openrouter(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "ENV_PATH", _write_env(tmp_path, "GOOGLE_API_KEY=AIzaReal\n"))
        with pytest.raises(SystemExit):
            config.require_openrouter_key()


class TestPlaceholderRejection:
    """An unedited .env must read as 'no key', not as a key that 401s."""

    @pytest.mark.parametrize("value", [
        "PASTE_YOUR_GEMINI_KEY_HERE",
        "SET_HF_USER",
        "your-key-here",
        "<your key>",
        "",
        "   ",
    ])
    def test_placeholders_are_not_treated_as_keys(self, tmp_path, monkeypatch, value):
        monkeypatch.setattr(config, "ENV_PATH", _write_env(tmp_path, f"GOOGLE_API_KEY={value}\n"))
        assert config.get_api_key("GOOGLE_API_KEY") is None

    def test_the_shipped_env_template_is_a_placeholder(self):
        """project.env.example must never contain a usable-looking key."""
        template = (config.REPO_ROOT / "project.env.example").read_text(encoding="utf-8")
        for line in template.splitlines():
            if line.startswith("GOOGLE_API_KEY="):
                assert line.split("=", 1)[1].strip() == "", "template must ship with an empty key"


class TestErrorMessage:
    def test_names_the_file_and_offers_the_offline_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "ENV_PATH", tmp_path / ".env")
        with pytest.raises(SystemExit) as excinfo:
            config.require_gemini_key()
        msg = str(excinfo.value)
        assert str(tmp_path / ".env") in msg      # tells you WHICH file
        assert "aistudio.google.com/apikey" in msg  # tells you where to get one
        assert "--teacher mock" in msg              # tells you how to proceed without one

    def test_succeeds_once_configured(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "ENV_PATH", _write_env(tmp_path, "GOOGLE_API_KEY=real-key\n"))
        assert config.require_gemini_key() == "real-key"
