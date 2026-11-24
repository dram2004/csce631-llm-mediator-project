"""TAMU AI gateway client, call log, and 24-hour budget-window guard.

Gateway behaviour handled here (TAMU-API-Guide.pdf and PA experience):
  * Two credentials: the shared course API key (TAMU_API_KEY) and the personal
    CF_Authorization cookie (CF_COOKIE), which expires about every 24 hours.
  * stream=True is required; the reply is reassembled from streamed chunks.
  * 1-second delay after every call.
  * Thinking-mode Claude models need temperature=1 and max_tokens >= 16384,
    and may inline <think>...</think> spans into the content; those are removed.
  * Identical requests return cached, byte-identical replies, so every call
    carries a fresh request ID (see prompts.nonce_line).
  * Reported usage omits most thinking tokens, so cost estimates for those
    models are scaled by ModelSpec.usage_underreport.

Credentials are never logged or printed.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import openai

from . import settings as S

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class TAMUAPIError(RuntimeError):
    pass


class RunStopped(RuntimeError):
    """Stop a run cleanly: credentials, budget, or a persistently failing gateway."""


def now() -> datetime:
    return datetime.now().astimezone()


# ---------------------------------------------------------------- client
def _credentials() -> tuple[str, str]:
    key = os.environ.get("TAMU_API_KEY", "").strip()
    cookie = os.environ.get("CF_COOKIE", "").strip()
    if not key:
        raise RunStopped("TAMU_API_KEY is not set. Add it to .env (see .env.example).")
    if not cookie:
        raise RunStopped("CF_COOKIE is not set. Log into https://chat.tamu.ai, copy the "
                         "CF_Authorization cookie (TAMU-API-Guide.pdf, Step 2) into .env.")
    if not cookie.startswith("CF_Authorization="):
        cookie = f"CF_Authorization={cookie}"
    return key, cookie


def _client() -> openai.OpenAI:
    key, cookie = _credentials()
    return openai.OpenAI(base_url=S.BASE_URL, api_key=key,
                         default_headers={"Cookie": cookie},
                         timeout=S.REQUEST_TIMEOUT_SECONDS)


def _detail(exc: openai.APIError) -> str:
    return f"HTTP {getattr(exc, 'status_code', '?')}; gateway body: {getattr(exc, 'body', None)!r}"


def list_models() -> list[str]:
    try:
        return sorted(m.id for m in _client().models.list())
    except openai.AuthenticationError as exc:
        raise RunStopped(f"Authentication failed ({_detail(exc)}). The CF cookie has "
                         "probably expired; copy a fresh one into .env.") from exc
    except openai.APIError as exc:
        raise RunStopped(f"Could not list models ({_detail(exc)}).") from exc


def call_model(spec: S.ModelSpec, messages: list[dict]) -> dict[str, Any]:
    """One streamed chat completion. Returns content, model version and usage."""
    if S.is_thinking_mode(spec.id) and (spec.temperature != 1.0
                                        or spec.max_tokens < S.MIN_THINKING_MODE_MAX_TOKENS):
        raise TAMUAPIError(f"{spec.id} is a thinking-mode model: needs temperature=1 "
                           f"and max_tokens >= {S.MIN_THINKING_MODE_MAX_TOKENS}.")
    client = _client()
    started = time.monotonic()
    try:
        stream = client.chat.completions.create(
            model=spec.id, messages=messages, temperature=spec.temperature,
            max_tokens=spec.max_tokens, stream=True,
            stream_options={"include_usage": True})
        chunks = list(stream)
    except openai.AuthenticationError as exc:
        raise RunStopped(f"Authentication failed ({_detail(exc)}). Unless the body says the "
                         "budget is exhausted, the CF cookie has expired: copy a fresh one "
                         "into .env and rerun.") from exc
    except openai.RateLimitError as exc:
        raise RunStopped(f"Rate limited or budget exhausted ({_detail(exc)}).") from exc
    except openai.APIConnectionError as exc:
        raise TAMUAPIError("Could not connect to the TAMU gateway.") from exc
    except openai.APIError as exc:
        body = str(getattr(exc, "body", "")).lower()
        if "budget" in body:
            raise RunStopped(f"Gateway reports the budget is exhausted ({_detail(exc)}).") from exc
        raise TAMUAPIError(f"Gateway error ({_detail(exc)}).") from exc
    finally:
        time.sleep(S.INTER_CALL_DELAY_SECONDS)

    parts, version, tin, tout = [], spec.id, 0, 0
    for ch in chunks:
        version = getattr(ch, "model", None) or version
        choices = getattr(ch, "choices", None)
        if choices:
            piece = getattr(choices[0].delta, "content", None)
            if piece:
                parts.append(piece)
        usage = getattr(ch, "usage", None)
        if usage is not None:
            tin = getattr(usage, "prompt_tokens", 0) or tin
            tout = getattr(usage, "completion_tokens", 0) or tout
    content = _THINK_RE.sub(" ", "".join(parts)).strip()
    if not content:
        raise TAMUAPIError("Empty response (content filter, or reasoning only).")
    return {"content": content, "model_version": version, "input_tokens": tin,
            "output_tokens": tout, "latency_s": round(time.monotonic() - started, 2)}


def estimated_cost(spec: S.ModelSpec, tin: int, tout: int) -> float:
    return (tin * spec.usd_per_m_in + tout * spec.usage_underreport * spec.usd_per_m_out) / 1e6


# ---------------------------------------------------------------- budget window
class BudgetLedger:
    """Every call attempt (timestamp, model, tokens, estimated cost) in
    results/calls.jsonl, including pilot calls, so the open budget window can
    be reconstructed across runs. A window opens at the first call and lasts
    24 hours; the first call after it ends opens the next one."""

    def __init__(self, path: Path):
        self.path = path
        self.rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []

    def add(self, row: dict):
        self.rows.append(row)
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def window(self) -> tuple[datetime | None, float]:
        start, spent = None, 0.0
        for r in self.rows:
            t = datetime.fromisoformat(r["ts"])
            if start is None or t >= start + S.WINDOW:
                start, spent = t, 0.0
            spent += r["cost_usd"]
        if start is None or now() >= start + S.WINDOW:
            return None, 0.0
        return start, spent

    def check(self, expected_next: float):
        start, spent = self.window()
        if start is not None and spent + expected_next > S.WINDOW_BUDGET_USD:
            raise RunStopped(
                f"Estimated ${spent:.2f} spent in the budget window that opened "
                f"{start:%a %b %d %H:%M}; the next call could pass ${S.WINDOW_BUDGET_USD:.2f}. "
                f"The window resets {start + S.WINDOW:%a %b %d %H:%M}. Rerun after that "
                "(with a fresh CF cookie) to resume.")

    def describe(self) -> str:
        start, spent = self.window()
        if start is None:
            return "No budget window open; the next call opens a new 24-hour window."
        return (f"Window opened {start:%a %b %d %H:%M}, resets {start + S.WINDOW:%a %b %d %H:%M}; "
                f"estimated ${spent:.2f} of ${S.DAILY_CAP_USD:.2f} spent "
                f"(guard stops at ${S.WINDOW_BUDGET_USD:.2f}).")

    def mean_cost(self, model_key: str, n: int = 30) -> float:
        xs = [r["cost_usd"] for r in self.rows if r["model_key"] == model_key][-n:]
        return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------- run log
class RunLog:
    """Experimental results, one JSON line per finished call_id. Rerunning
    skips every call_id that already has a record."""

    def __init__(self, path: Path):
        self.path = path
        self.done: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                self.done[r["call_id"]] = r

    def append(self, rec: dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.done[rec["call_id"]] = rec


def extract_json(text: str) -> dict:
    """The last JSON object in a reply (tolerates prose and ```json fences)."""
    s = text.strip()
    end = s.rfind("}")
    if end < 0:
        raise ValueError("no JSON object found")
    depth = 0
    for i in range(end, -1, -1):
        depth += s[i] == "}"
        depth -= s[i] == "{"
        if depth == 0:
            return json.loads(s[i:end + 1])
    raise ValueError("unbalanced JSON")


def run_call(ledger: BudgetLedger, log: RunLog, call_id: str, spec: S.ModelSpec,
             build_messages, parse, meta: dict) -> dict:
    """Run one experimental call (skipped if already logged).

    `build_messages(nonce)` returns the messages with a unique request ID, so a
    re-ask is never answered from the gateway's cache. A reply that cannot be
    parsed is re-asked up to PARSE_ATTEMPTS times; every reply is kept."""
    if call_id in log.done:
        return log.done[call_id]
    import uuid
    replies = []
    parsed = None
    for _ in range(S.PARSE_ATTEMPTS):
        ledger.check(ledger.mean_cost(spec.key))
        nonce = uuid.uuid4().hex[:12]
        res = None
        for attempt in range(1, S.CALL_ATTEMPTS + 1):
            try:
                res = call_model(spec, build_messages(nonce))
                break
            except TAMUAPIError as exc:
                if attempt == S.CALL_ATTEMPTS:
                    raise RunStopped(f"Gateway kept failing after {attempt} tries: {exc}") from exc
                print(f"    gateway error ({exc}); retrying in {S.RETRY_BACKOFF_SECONDS:.0f}s")
                time.sleep(S.RETRY_BACKOFF_SECONDS)
        cost = estimated_cost(spec, res["input_tokens"], res["output_tokens"])
        ledger.add({"ts": now().isoformat(), "model_key": spec.key, "call_id": call_id,
                    "input_tokens": res["input_tokens"], "output_tokens": res["output_tokens"],
                    "cost_usd": cost})
        reply = {"nonce": nonce, "raw": res["content"], "model_version": res["model_version"],
                 "input_tokens": res["input_tokens"], "output_tokens": res["output_tokens"],
                 "latency_s": res["latency_s"], "cost_usd": cost}
        try:
            parsed = parse(res["content"])
            reply["parse_error"] = None
            replies.append(reply)
            break
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            reply["parse_error"] = str(e)
            replies.append(reply)
    rec = {"call_id": call_id, "ok": parsed is not None, "model_key": spec.key,
           "model_id": spec.id, "ts": now().isoformat(), **meta, "parsed": parsed,
           "n_attempts": len(replies), "replies": replies}
    log.append(rec)
    return rec