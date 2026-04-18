import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Config:
    api_key: str
    model: str = "glm-4.7"
    base_url: str | None = None
    max_tokens: int = 8000
    bash_timeout: int = 120
    nag_threshold: int = 3


def load_config() -> Config:
    """Load config from project .env file."""
    # Load .env from the package's project root (flux/../.env)
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    api_key = os.environ.get("FLUX_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "Error: FLUX_API_KEY not found\n"
            "  echo 'FLUX_API_KEY=your-key' > .env"
        )
    return Config(
        api_key=api_key,
        model=os.environ.get("FLUX_MODEL", "glm-4.7"),
        base_url=os.environ.get("FLUX_BASE_URL"),
        max_tokens=int(os.environ.get("FLUX_MAX_TOKENS", "8000")),
        bash_timeout=int(os.environ.get("FLUX_BASH_TIMEOUT", "120")),
        nag_threshold=int(os.environ.get("FLUX_NAG_THRESHOLD", "3")),
    )
