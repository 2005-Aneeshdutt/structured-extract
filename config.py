"""Project configuration: load `.env` and fetch credentials with useful errors.

Why this module exists at all
-----------------------------
`python-dotenv` was in requirements.txt from the start, but nothing ever called
`load_dotenv()`. The result would have been the worst kind of setup bug: you
create `.env`, paste your key in, run `make label-gold`, and get
"GOOGLE_API_KEY not set" -- with no hint that the file you just edited is never
read. One call site, imported by every entry point that needs a credential.

Precedence is deliberate: a real environment variable always beats `.env`, so
CI and Kaggle (which inject secrets as env vars) work without a file, and a
one-off `GOOGLE_API_KEY=... make label-gold` overrides the file for a single run.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
ENV_PATH = REPO_ROOT / ".env"

_loaded = False


def load_project_env(path: Path | None = None) -> bool:
    """Load `.env` into os.environ if present. Idempotent. Returns True if read.

    Existing environment variables are NOT overwritten -- see the module
    docstring for why that precedence matters.

    Falls back to a minimal hand-rolled parser when python-dotenv is missing, so
    CI (which installs a reduced dependency set) does not need the package just
    to run a smoke test.
    """
    global _loaded
    if _loaded:
        return True
    env_path = path or ENV_PATH
    if not env_path.exists():
        return False

    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        # KEY=value, ignoring blanks/comments and optional surrounding quotes.
        # Deliberately not a full dotenv implementation -- no interpolation, no
        # multiline values; the real parser handles those when installed.
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if key and value and key not in os.environ:
                os.environ[key] = value

    _loaded = True
    LOGGER.debug("loaded environment from %s", env_path)
    return True


def get_api_key(*names: str) -> str | None:
    """First non-empty value among `names`, after loading `.env`."""
    load_project_env()
    for name in names:
        value = (os.environ.get(name) or "").strip()
        # Guard against the template placeholders being left in place -- an
        # unedited .env should read as "no key", not as a key that 401s.
        if value and not value.startswith(("PASTE_", "SET_", "your-", "<")):
            return value
    return None


def require_gemini_key() -> str:
    """Gemini API key, or exit with instructions that name the actual file."""
    key = get_api_key("GOOGLE_API_KEY", "GEMINI_API_KEY")
    if key:
        return key
    have_env = ENV_PATH.exists()
    raise SystemExit(
        "No Gemini API key found.\n\n"
        + (
            f"  {ENV_PATH} exists but GOOGLE_API_KEY is empty or still a placeholder.\n"
            "  Open it and paste your key after 'GOOGLE_API_KEY='.\n"
            if have_env
            else f"  Create {ENV_PATH}:\n"
                 f"      cp project.env.example .env\n"
                 "  then paste your key after 'GOOGLE_API_KEY='.\n"
        )
        + "\n  Free key (no billing account needed): https://aistudio.google.com/apikey\n"
        "\n  To try the pipeline with no key at all, use the offline teacher:\n"
        "      python -m data.generate_synthetic --n 200 --teacher mock\n"
    )


def get_hf_token() -> str | None:
    """HF token for pushing models/datasets. Optional -- reads are anonymous."""
    return get_api_key("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
