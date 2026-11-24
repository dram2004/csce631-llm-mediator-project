"""Single entry point for all live API work.

  python run.py verify             # check the pinned model IDs against the gateway's list
  python run.py pilot              # 1 call per model per Part A cell and private Part B cell
  python run.py adopt-pilot        # count the pilot's calls as sample 0 of the real runs
  python run.py partA              # Part A: 9 cells x 3 models x 5 samples
  python run.py partB              # Part B private (core): 20 cells x 3 models x 10 samples
  python run.py partB --public     # Part B public (H3 extension): 16 cells x 3 models x 5 samples
  python run.py budget             # open budget window, spend, and projected cost of what is left

Options: --models gpt55,sonnet restricts models. Every run resumes: a call_id
already in results/ is never repeated. Runs stop on their own before the
estimated spend in the open 24-hour budget window passes WINDOW_BUDGET_USD.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np

from mediator import design, prompts, solvers as Sv
from mediator import settings as S
from mediator.games import GAMES
from mediator.llm import BudgetLedger, RunLog, RunStopped, list_models, run_call


def ledger() -> BudgetLedger:
    S.RESULTS_DIR.mkdir(exist_ok=True)
    return BudgetLedger(S.RESULTS_DIR / "calls.jsonl")


def models_from(args) -> list[S.ModelSpec]:
    return [S.MODELS[k] for k in (args.models or list(S.MODELS))]


# ---------------------------------------------------------------- verify
def cmd_verify(args) -> None:
    available = set(list_models())
    ok = True
    for spec in models_from(args):
        found = spec.id in available
        ok &= found
        mode = "thinking mode" if S.is_thinking_mode(spec.id) else "standard"
        print(f"  {'OK     ' if found else 'MISSING'} {spec.key:7s} {spec.id!r:34s} {mode}")
    if not ok:
        similar = [m for m in sorted(available)
                   if any(w in m.lower() for w in ("gpt-5", "gpt-4o", "sonnet"))]
        print("\nClosest IDs on the gateway (set MODEL_GPT55 / MODEL_GPT4O / MODEL_SONNET "
              "in .env to one of these):\n  " + "\n  ".join(similar))
        sys.exit(1)
    print("All pinned model IDs are available.")


# ---------------------------------------------------------------- call plans
def plan_a(args, samples: int) -> list[tuple]:
    return [(f"A|{c['game']}|{c['objective']}|{m.key}|{s}", c, m, s)
            for c in design.part_a_cells() for m in models_from(args) for s in range(samples)]


def plan_b(args, condition: str, samples: int) -> list[tuple]:
    return [(f"B|{c['game']}|{c['condition']}|{c['level']}|p{c['player']}|r{c['rec']}|{m.key}|{s}",
             c, m, s)
            for c in design.part_b_cells(condition) for m in models_from(args)
            for s in range(samples)]


# ---------------------------------------------------------------- Part A
def part_a(args, out: str = "partA.jsonl", samples: int | None = None) -> None:
    led, log = ledger(), RunLog(S.RESULTS_DIR / out)
    plan = plan_a(args, samples or S.PART_A["samples"])
    left = [p for p in plan if p[0] not in log.done]
    print(f"Part A: {len(plan)} calls, {len(left)} still to run. {led.describe()}")
    for cid, c, m, s in left:
        g = GAMES[c["game"]]
        rec = run_call(led, log, cid, m,
                       lambda nonce: prompts.mediator_messages(g, c["objective"], nonce),
                       prompts.parse_device(g), meta={"part": "A", **c, "sample": s})
        status = "ok" if rec["ok"] else "UNPARSED"
        print(f"  {cid:42s} {status:8s} ${sum(r['cost_usd'] for r in rec['replies']):.4f}")


# ---------------------------------------------------------------- Part B
def part_b(args, condition: str, out: str = "partB.jsonl", samples: int | None = None) -> None:
    led, log = ledger(), RunLog(S.RESULTS_DIR / out)
    n = samples or S.PART_B[f"{condition}_samples"]
    plan = plan_b(args, condition, n)
    left = [p for p in plan if p[0] not in log.done]
    print(f"Part B ({condition}): {len(plan)} calls, {len(left)} still to run. {led.describe()}")
    for cid, c, m, s in left:
        g = GAMES[c["game"]]
        mu = np.array(c["mu"])
        p, rec_a = c["player"], c["rec"]
        opp_rec, delta = None, c["delta"]
        if condition == "public":
            # opponent's announced recommendation, drawn from the device; fixed per call_id
            rng = np.random.default_rng(int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16))
            post = Sv.conditional_opponent(mu, p, rec_a)
            opp_rec = int(rng.choice(len(post), p=post))
            U = g.payoffs(p)
            col = U[:, opp_rec] if p == 0 else U[opp_rec, :]
            delta = float(max(col[a] for a in range(len(col)) if a != rec_a) - col[rec_a])
        meta = {"part": "B", **{k: v for k, v in c.items() if k != "mu"},
                "delta": delta, "delta_private": c["delta"], "opp_rec": opp_rec, "sample": s}
        rec = run_call(led, log, cid, m,
                       lambda nonce: prompts.player_messages(g, mu, p, rec_a, condition,
                                                             opp_rec, nonce),
                       prompts.parse_action(g, p), meta=meta)
        act = rec["parsed"]["action"] if rec["ok"] else None
        obey = "UNPARSED" if act is None else ("obey" if act == rec_a else "DEVIATE")
        print(f"  {cid:44s} Δ={delta:+.2f}  {obey:8s} "
              f"${sum(r['cost_usd'] for r in rec['replies']):.4f}")


# ---------------------------------------------------------------- pilot
def cmd_pilot(args) -> None:
    cmd_verify(args)
    part_a(args, out="pilot_partA.jsonl", samples=1)
    part_b(args, "private", out="pilot_partB.jsonl", samples=1)
    report(args)


def cmd_adopt_pilot(args) -> None:
    """The pilot used the final prompts and the same call IDs (sample 0), so its
    calls are valid sample-0 data. Copy any not already present."""
    for src, dst in (("pilot_partA.jsonl", "partA.jsonl"), ("pilot_partB.jsonl", "partB.jsonl")):
        sp, dp = S.RESULTS_DIR / src, S.RESULTS_DIR / dst
        if not sp.exists():
            print(f"  {src}: not found, skipped")
            continue
        have = RunLog(dp).done
        new = [l for l in sp.read_text().splitlines()
               if l.strip() and json.loads(l)["call_id"] not in have]
        with open(dp, "a") as f:
            for l in new:
                f.write(l + "\n")
        print(f"  {src} -> {dst}: {len(new)} calls adopted as sample 0")


# ---------------------------------------------------------------- budget report
def report(args) -> None:
    led = ledger()
    print("\n=== budget ===")
    print(led.describe())
    if not led.rows:
        print("No calls logged yet.")
        return
    # mean estimated cost per call, by part and model (Part A replies are much longer)
    per: dict[tuple[str, str], list[float]] = {}
    for r in led.rows:
        per.setdefault((r["call_id"][0], r["model_key"]), []).append(r["cost_usd"])
    print(f"{'part':5s} {'model':8s} {'calls':>6s} {'est $/call':>11s}")
    for (part, mk), xs in sorted(per.items()):
        print(f"{part:5s} {mk:8s} {len(xs):6d} {np.mean(xs):11.5f}")

    def cost_left(plan, log_name, part):
        done = RunLog(S.RESULTS_DIR / log_name).done
        left = [p for p in plan if p[0] not in done]
        usd = sum(np.mean(per.get((part, p[2].key), [0.0])) for p in left)
        return len(left), usd

    print("\nStill to run (estimated; Sonnet output scaled for under-reported thinking tokens):")
    total = 0.0
    for name, plan, log_name, part in (
            ("Part A", plan_a(args, S.PART_A["samples"]), "partA.jsonl", "A"),
            ("Part B private", plan_b(args, "private", S.PART_B["private_samples"]), "partB.jsonl", "B"),
            ("Part B public", plan_b(args, "public", S.PART_B["public_samples"]), "partB.jsonl", "B")):
        n, usd = cost_left(plan, log_name, part)
        total += usd
        print(f"  {name:15s} {n:5d} calls  ≈ ${usd:6.2f}")
    print(f"  {'total':15s}        ≈ ${total:6.2f}  ≈ {total / S.WINDOW_BUDGET_USD:.1f} "
          f"budget windows at ${S.WINDOW_BUDGET_USD:.2f}")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["verify", "pilot", "adopt-pilot", "partA", "partB", "budget"])
    ap.add_argument("--public", action="store_true", help="Part B public condition (H3)")
    ap.add_argument("--models", type=lambda s: s.split(","))
    args = ap.parse_args()
    try:
        if args.cmd == "verify":
            cmd_verify(args)
        elif args.cmd == "pilot":
            cmd_pilot(args)
        elif args.cmd == "adopt-pilot":
            cmd_adopt_pilot(args)
        elif args.cmd == "partA":
            part_a(args)
        elif args.cmd == "partB":
            part_b(args, "public" if args.public else "private")
        if args.cmd in ("adopt-pilot", "partA", "partB", "budget"):
            report(args)
    except RunStopped as exc:
        print(f"\nStopped cleanly: {exc}\nEverything finished so far is saved; rerun the "
              "same command to resume.")
        report(args)
    except KeyboardInterrupt:
        print("\nInterrupted. Everything finished so far is saved; rerun to resume.")


if __name__ == "__main__":
    main()