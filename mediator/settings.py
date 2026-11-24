"""Experiment settings.

Credentials and budget knobs come from .env (see .env.example); the
experimental design lives here so it is versioned with the code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------- gateway
BASE_URL: str = os.environ.get("TAMU_BASE_URL", "https://chat.tamu.ai/api")
INTER_CALL_DELAY_SECONDS: float = 1.0          # course guide: 1 s between calls
REQUEST_TIMEOUT_SECONDS: float = 120.0
CALL_ATTEMPTS: int = 3                         # tries per call on transient gateway errors
RETRY_BACKOFF_SECONDS: float = 5.0
PARSE_ATTEMPTS: int = 3                        # 1 call + 2 fresh re-asks if the reply cannot be parsed

# ---------------------------------------------------------------- budget
# $5 per student per 24-hour window. The window opens at the first API call
# and resets exactly 24 hours later; the next call after that opens a new one.
DAILY_CAP_USD: float = 5.00
WINDOW: timedelta = timedelta(hours=24)
# Stop before the estimated spend in the open window would pass this.
WINDOW_BUDGET_USD: float = float(os.environ.get("WINDOW_BUDGET_USD", "4.50"))


@dataclass(frozen=True)
class ModelSpec:
    key: str
    id: str                    # exact gateway identifier (case and spacing matter)
    role: str
    temperature: float
    max_tokens: int
    usd_per_m_in: float        # list price, input
    usd_per_m_out: float       # list price, output
    usage_underreport: float   # multiply reported output tokens by this for cost estimates


# Every model runs at temperature 1.0: thinking-mode Claude models require it,
# and using it everywhere keeps sampling noise comparable across models.
# Reasoning models get a large max_tokens because hidden reasoning counts
# against it (billing is for tokens produced, not the cap).
# Prices are list-price ESTIMATES used only by the budget guard; the gateway's
# own $5 cap is the real limit. Check them against the pilot report.
MODELS: dict[str, ModelSpec] = {
    "gpt55": ModelSpec("gpt55", os.environ.get("MODEL_GPT55", "protected.gpt-5.5"),
                       "frontier reasoning", 1.0, 16384, 5.00, 30.00, 1.0),
    "gpt4o": ModelSpec("gpt4o", os.environ.get("MODEL_GPT4O", "protected.gpt-4o"),
                       "non-reasoning baseline", 1.0, 512, 2.50, 10.00, 1.0),
    "sonnet": ModelSpec("sonnet", os.environ.get("MODEL_SONNET", "protected.Claude-Sonnet-5"),
                        "frontier, second provider", 1.0, 16384, 3.00, 15.00, 4.0),
}


def is_thinking_mode(model_id: str) -> bool:
    """Claude Sonnet and Opus run in thinking mode on this gateway
    (temperature must be 1, max_tokens >= 16384)."""
    s = model_id.lower()
    return "claude" in s and ("sonnet" in s or "opus" in s)


MIN_THINKING_MODE_MAX_TOKENS: int = 16384

# ---------------------------------------------------------------- design
SEED: int = 631
RESULTS_DIR: Path = ROOT / "results"

# Scope after the pilot (Oct 29): Part A at 5 samples, Part B on the two 2x2
# games at 4 temptation levels. The pilot's calls are adopted as sample 0
# (same prompts, same call IDs), so they are not paid for twice.
PART_A = {
    "games": ["chicken", "bos", "shapley"],
    "objectives": ["welfare", "egalitarian", "p1"],
    "samples": 5,
}

# Temptation is indexed by the deviation gain Delta (payoff units), not the
# mixing weight lambda: design.py solves for the lambda that hits each target.
PART_B = {
    "games": ["chicken", "bos"],
    "delta_targets": {"chicken": [-1.5, -0.25, 0.25, 1.0], "bos": [-2.0, -0.25, 0.25, 2.0]},
    "private_samples": 10,     # core: RQ2 / H2
    "public_samples": 5,       # extension: H3, focal recommendations only
}

# Cut first to last if the budget binds.
DROP_ORDER = [
    "public condition (H3)",
    "Part B private samples 10 -> 6",
    "Part A samples 5 -> 3 for the two frontier models (both at ceiling in the pilot)",
]