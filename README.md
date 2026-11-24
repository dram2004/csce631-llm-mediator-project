# Can an LLM Be a Mediator? — experiment code

CSCE 631-D course project. Every LLM output is scored automatically against
exact correlated-equilibrium (LP) and Nash references; nothing is checked by hand.

## Setup

```bash
cd mediator_ce
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env             # then fill in TAMU_API_KEY and CF_COOKIE
python -m pytest -q tests        # 20 solver/checker tests should pass
```

`CF_COOKIE` expires about every 24 hours: before each session, log into
https://chat.tamu.ai and copy a fresh `CF_Authorization` value into `.env`
(TAMU-API-Guide.pdf, Step 2).

## Run order

| Step | Command | Calls | What it does |
|---|---|---|---|
| 1 | `python run.py verify` | 0 | Confirms the three model IDs exist on the gateway |
| 2 | `python run.py pilot` | 87 | 1 call per model per Part A cell and private Part B cell (done Oct 29) |
| 3 | `python run.py adopt-pilot` | 0 | Counts the pilot's calls as sample 0 of the real runs |
| 4 | `python run.py partA` | 135 (108 new) | Part A: 9 cells × 3 models × 5 samples |
| 5 | `python run.py partB` | 600 (540 new) | Part B private (core, H2): 20 cells × 3 models × 10 samples |
| 6 | `python run.py partB --public` | 240 | Part B public (H3): 16 tempted cells × 3 models × 5 samples |
| any | `python analyze.py` | 0 | Rebuilds every table and figure in `results/analysis/` (`--results <folder>` to point elsewhere) |
| any | `python run.py budget` | 0 | Open window, spend, and estimated cost of everything still to run |

Steps 4–6 each stop on their own at the window's budget and resume after the reset.

## Budget

The gateway caps each student at **$5 per 24-hour window**. The window opens
at your first API call and resets exactly 24 hours later (first call at
11:00 → reset at 11:00 the next day); the first call after that opens a new
window. Every call is logged to `results/calls.jsonl`, and runs stop on their
own before the estimated spend in the open window passes `WINDOW_BUDGET_USD`
(default $4.50), printing the reset time. Rerun the same command after the
reset, with a fresh cookie, to resume; finished calls are never repeated.

Estimates use list prices in `mediator/settings.py`; reported usage omits most
Claude thinking tokens, so Sonnet's output is scaled up for the estimate. The
gateway's own cap is the real limit: a budget error from it also stops the run.

## Gateway handling (`mediator/llm.py`)

* Base URL `https://chat.tamu.ai/api`, API key plus `CF_Authorization` cookie header.
* `stream=True` on every call (non-streamed replies are malformed on this gateway).
* 1-second pause after every call; transient errors retried 3 times.
* Thinking-mode Claude models: temperature 1 and `max_tokens` ≥ 16384; `<think>` spans removed.
  All three models run at temperature 1 so sampling noise is comparable.
* The gateway caches identical requests, so every call (including re-asks of an
  unparseable reply) carries a unique request ID.

## Layout

```
.env.example           credentials and budget knobs (copy to .env)
mediator/settings.py   model IDs and parameters, budget window, experimental design, drop order
mediator/games.py      Chicken, Battle of the Sexes, Shapley's game
mediator/solvers.py    CE LP (welfare / egalitarian / player 1), Nash enumeration,
                       obedience checker, distance-to-CE LP, deviation gain Delta
mediator/design.py     call grid; solves lambda for each target Delta
mediator/prompts.py    mediator and player prompts + strict JSON parsers
mediator/llm.py        TAMU client, result log, budget-window ledger
run.py / analyze.py    live runs / automated scoring, tables, figures
```

## Reproducing the final report

The logged results in `results/` are the data behind every number in the final
report. Rebuilding the tables and figures needs no API access and is deterministic:

```bash
python -m pytest -q tests
python analyze.py --results results
```

Key outputs in `results/analysis/`:

| File | Report |
|---|---|
| `partA_by_model_game.csv`, `partA_by_model_objective.csv`, `partA_H1.csv` | Table 3, Table 6, H1 |
| `random_device_baseline.csv` | chance baseline |
| `partB_H2_by_level.csv`, `obedience_by_delta.pdf` | Figure 1 |
| `partB_H2_by_regime.csv`, `partB_H2_matched.csv` | Table 4 (H2) |
| `partB_H3_prereg.csv`, `partB_H3_vs_bench.csv` | Table 5 (H3) |
| `partB_nonrational_reasons.csv` | stated reasons (Discussion, Appendix C) |

The `results/` logs contain prompts' request IDs, replies and token counts; they
contain no credentials (`.env` is never committed).
