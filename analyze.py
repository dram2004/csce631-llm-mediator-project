"""Automated scoring of every logged LLM output against the LP / Nash references.

  python analyze.py              # scores everything in results/
  python analyze.py --pilot      # scores the pilot files instead

Writes CSV tables and PNG figures into results/analysis/ (or results/analysis_pilot/). Nothing is graded
by hand: every device is checked against all obedience constraints by
mediator.solvers, and every player decision is compared with the exact
deviation gain of its cell.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, norm

from mediator.games import GAMES
from mediator import solvers as S
from mediator import settings as SET

ROOT = Path(__file__).resolve().parent
COLORS = {"gpt55": "#2a78d6", "gpt4o": "#eb6834", "sonnet": "#1baf7a"}
MARKERS = {"gpt55": "o", "gpt4o": "s", "sonnet": "^"}
LABELS = {"gpt55": "GPT-5.5", "gpt4o": "GPT-4o", "sonnet": "Claude Sonnet 5"}
EPS_CE = 0.01   # epsilon-CE tolerance: 1% of the game's payoff range
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    df = pd.DataFrame(rows).drop_duplicates("call_id", keep="last")
    return df.reset_index(drop=True)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


# ------------------------------------------------------------------ Part A
def score_part_a(df: pd.DataFrame) -> pd.DataFrame:
    refs = {}
    out = []
    for _, r in df.iterrows():
        g = GAMES[r["game"]]
        obj = r["objective"]
        key = (r["game"], obj)
        if key not in refs:
            lp = S.solve_ce(g, obj)["value"]
            ne = S.objective_value(g, S.public_nash_mix(g, obj)["mu"], obj)
            refs[key] = (lp, ne)
        lp, ne = refs[key]
        rec = {"model": r["model_key"], "game": r["game"], "objective": obj,
               "parsed": bool(r["ok"])}
        if r["ok"]:
            p = r["parsed"]
            raw = np.array(p["device_raw"])
            mu = (raw / raw.sum()).reshape(g.shape)
            rep = S.check_obedience(g, mu)
            val = S.objective_value(g, mu, obj)
            rec.update({
                "sums_to_one": p["sums_to_one"],
                "is_ce": rep.is_ce,
                "n_violated": rep.n_violated,
                "max_violation_norm": rep.max_violation_norm,
                "dist_to_ce": S.distance_to_ce(g, mu),
                "attainment": val / lp if lp else np.nan,
                "gain_captured": (val - ne) / (lp - ne) if lp - ne > 1e-9 else np.nan,
                "valid_and_ce": bool(rep.is_ce and p["sums_to_one"]),
                # tolerant score: rounding to 2-3 decimals (1/6 -> 0.167) should not count
                "valid_and_eps_ce": bool(p["sums_to_one"] and rep.max_violation_norm <= EPS_CE),
            })
        out.append(rec)
    return pd.DataFrame(out)


def random_baseline(k: int = 20000, seed: int = 631) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in GAMES.items():
        devs = S.random_devices(g, k, rng)
        ce = np.array([S.check_obedience(g, mu).is_ce for mu in devs])
        welf = np.array([S.objective_value(g, mu, "welfare") for mu in devs])
        best_ne = S.objective_value(g, S.best_nash(g, "welfare")["mu"], "welfare")
        rows.append({"game": key, "random_ce_rate": ce.mean(),
                     "random_beats_best_ne_welfare": (welf > best_ne + 1e-9).mean(),
                     "random_ce_and_beats_ne": (ce & (welf > best_ne + 1e-9)).mean()})
    return pd.DataFrame(rows)


def part_a_tables(sc: pd.DataFrame) -> dict:
    t = {}
    t["by_model_objective"] = (sc.groupby(["model", "objective"]).agg(
        n=("parsed", "size"), parse_rate=("parsed", "mean"),
        ce_rate=("valid_and_ce", "mean"), eps_ce_rate=("valid_and_eps_ce", "mean"),
        max_viol=("max_violation_norm", "mean"),
        attainment=("attainment", "mean"), gain_captured=("gain_captured", "mean"),
        dist_to_ce=("dist_to_ce", "mean")).round(3))
    t["by_model_game"] = (sc.groupby(["model", "game"]).agg(
        ce_rate=("valid_and_ce", "mean"), attainment=("attainment", "mean")).round(3))
    # H1: violation rate under the self-favouring objective vs the other two
    h1 = []
    for m, d in sc.groupby("model"):
        bad = ~d["valid_and_eps_ce"].fillna(False)
        sp = d["objective"] == "p1"
        tab = [[bad[sp].sum(), (~bad[sp]).sum()], [bad[~sp].sum(), (~bad[~sp]).sum()]]
        odds, p = fisher_exact(tab, alternative="greater")
        h1.append({"model": m, "viol_rate_p1": bad[sp].mean(),
                   "viol_rate_welfare_egal": bad[~sp].mean(), "fisher_p_one_sided": p})
    t["H1"] = pd.DataFrame(h1).round(4)
    return t


# ------------------------------------------------------------------ Part B
def score_part_b(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["ok"]].copy()
    d["action"] = d["parsed"].apply(lambda p: p["action"])
    d["obey"] = (d["action"] == d["rec"]).where(d["condition"] != "none")
    d["model"] = d["model_key"]
    return d


def h2_tables(d: pd.DataFrame) -> dict:
    """H2 (over-obedience) on private decisions, without assuming a monotone curve.

    by_level:  obedience per (model, game, Delta level), both players pooled.
    by_regime: obedience per model for Delta < 0 and Delta > 0, with Delta > 0 split
               into stochastic devices (lambda < 1) and deterministic ones (lambda = 1,
               a point mass that reveals the opponent's recommendation).
    matched:   the matched-magnitude test. Chicken and BoS both have levels at
               Delta = -0.25 and +0.25, so the error when following pays (deviating
               at -0.25) can be compared with the error when deviating pays (obeying
               at +0.25). Symmetric noise predicts equal error rates; H2 predicts more
               errors at +0.25 (a bias toward the recommendation). One-sided Fisher test.
    """
    p = d[d["condition"] == "private"].copy()
    p["det"] = p["lam"].astype(float) >= 1 - 1e-9
    rows = []
    for (m, g, dv), x in p.assign(delta_private=p["delta_private"].round(2)).groupby(["model", "game", "delta_private"]):
        k, n = int(x["obey"].sum()), len(x)
        lo, hi = wilson(k, n)
        rows.append({"model": m, "game": g, "delta": round(dv, 2), "deterministic": bool(x["det"].iloc[0]),
                     "obey": k, "n": n, "rate": k / n, "ci_lo": lo, "ci_hi": hi})
    by_level = pd.DataFrame(rows)
    reg = []
    for m, x in p.groupby("model"):
        for name, mask in (("delta<0 (should obey)", x["delta_private"] < 0),
                           ("delta>0 stochastic (should deviate)", (x["delta_private"] > 0) & ~x["det"]),
                           ("delta>0 deterministic (should deviate)", (x["delta_private"] > 0) & x["det"])):
            k, n = int(x[mask]["obey"].sum()), int(mask.sum())
            lo, hi = wilson(k, n)
            reg.append({"model": m, "regime": name, "obey": k, "n": n, "rate": k / n,
                        "ci_lo": lo, "ci_hi": hi})
    by_regime = pd.DataFrame(reg)
    mt = []
    for m, x in p.groupby("model"):
        neg = x[np.isclose(x["delta_private"], -0.25, atol=0.01)]
        pos = x[np.isclose(x["delta_private"], 0.25, atol=0.01)]
        e_neg, e_pos = int((~neg["obey"].astype(bool)).sum()), int(pos["obey"].sum())
        _, pval = fisher_exact([[e_pos, len(pos) - e_pos], [e_neg, len(neg) - e_neg]],
                               alternative="greater")
        mt.append({"model": m, "err_at_-0.25 (deviated)": e_neg, "n_neg": len(neg),
                   "err_at_+0.25 (obeyed)": e_pos, "n_pos": len(pos), "fisher_p_one_sided": pval})
    return {"by_level": by_level, "by_regime": by_regime, "matched": pd.DataFrame(mt)}


def reasons_table(d: pd.DataFrame) -> pd.DataFrame:
    """Decisions that disagree with the rational benchmark, with the model's stated reason."""
    x = d[d["condition"] != "none"].copy()
    x["rational"] = np.where(x["delta"] > 1e-9, "deviate",
                             np.where(x["delta"] < -1e-9, "obey", "indifferent"))
    x["chose"] = np.where(x["obey"].astype(bool), "obey", "deviate")
    x = x[(x["rational"] != "indifferent") & (x["rational"] != x["chose"])]
    x["reason"] = x["parsed"].apply(lambda p: p.get("reason", ""))
    return x[["model", "game", "condition", "level", "player", "rec", "delta", "chose",
              "rational", "reason"]].sort_values(["model", "game", "delta"])


def obedience_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, g, cond, lvl, p, rec), x in d[d["condition"] != "none"].groupby(
            ["model", "game", "condition", "level", "player", "rec"]):
        k, n = int(x["obey"].sum()), len(x)
        lo, hi = wilson(k, n)
        rows.append({"model": m, "game": g, "condition": cond, "level": lvl,
                     "player": p, "rec": rec, "delta": x["delta_private"].iloc[0],
                     "obey_rate": k / n, "ci_lo": lo, "ci_hi": hi, "n": n})
    return pd.DataFrame(rows)


def h3_tables(d: pd.DataFrame) -> dict:
    """H3: does a public announcement lower obedience to non-Nash profiles?

    prereg:    the test as proposed. Obedience when the announced profile is not a
               Nash equilibrium (public Delta > 0) vs obedience in the same cells under
               private recommendation. Note: a rational player also obeys less here,
               because the announcement reveals that deviating pays.
    vs_bench:  obedience in each information regime next to its rational benchmark,
               split by stochastic (lambda < 1) vs deterministic (lambda = 1) devices.
    The opponent's announced action is drawn per call, so the mix of NE and non-NE
    announcements differs slightly across models; n is reported for every cell.
    """
    out = []
    pub = d[(d["condition"] == "public") & (d["delta"] > 1e-9)]
    for m, x in pub.groupby("model"):
        keys = set(zip(x["game"], x["level"], x["player"], x["rec"]))
        priv = d[(d["condition"] == "private") & (d["model"] == m)]
        priv = priv[[k in keys for k in zip(priv["game"], priv["level"], priv["player"], priv["rec"])]]
        k1, n1, k2, n2 = int(x["obey"].sum()), len(x), int(priv["obey"].sum()), len(priv)
        _, pval = fisher_exact([[k2, n2 - k2], [k1, n1 - k1]], alternative="greater")
        out.append({"model": m, "obey_public_nonNE": k1, "n_public": n1,
                    "obey_private_same_cells": k2, "n_private": n2, "fisher_p_one_sided": pval})
    prereg = pd.DataFrame(out)
    rows = []
    for m, x in d.groupby("model"):
        x = x.assign(det=x["lam"].astype(float) >= 1 - 1e-9)
        for cond in ("private", "public"):
            for sign, name in ((-1, "should obey"), (1, "should deviate")):
                for det in (False, True):
                    y = x[(x["condition"] == cond) & (np.sign(x["delta"]) == sign) & (x["det"] == det)]
                    if len(y) == 0:
                        continue
                    k, n = int(y["obey"].sum()), len(y)
                    lo, hi = wilson(k, n)
                    rows.append({"model": m, "condition": cond, "benchmark": name,
                                 "device": "deterministic" if det else "stochastic",
                                 "obey": k, "n": n, "rate": k / n, "ci_lo": lo, "ci_hi": hi})
    return {"prereg": prereg, "vs_bench": pd.DataFrame(rows)}


# ------------------------------------------------------------------ figures
def plot_obedience(lv: pd.DataFrame, out: Path, title_note: str = ""):
    """Private obedience by deviation gain, both players pooled per level (n = 20 per point,
    except the constant-Delta anchors). Hollow markers: deterministic devices (lambda = 1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    games = [("chicken", "Chicken"), ("bos", "Battle of the Sexes")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    offs = {"gpt55": -0.045, "gpt4o": 0.0, "sonnet": 0.045}
    for ax, (gk, title) in zip(axes, games):
        x = lv[lv["game"] == gk]
        span = x["delta"].max() - x["delta"].min()
        for m in ("gpt55", "gpt4o", "sonnet"):
            xm = x[x["model"] == m].sort_values("delta")
            for det in (False, True):
                xx = xm[xm["deterministic"] == det]
                if xx.empty:
                    continue
                xs = xx["delta"] + offs[m] * span
                ax.errorbar(xs, xx["rate"], yerr=[xx["rate"] - xx["ci_lo"], xx["ci_hi"] - xx["rate"]],
                            fmt=MARKERS[m], ms=5.5, color=COLORS[m], ecolor=COLORS[m],
                            mfc="white" if det else COLORS[m], mew=1.4, elinewidth=1, capsize=0,
                            lw=0, label=LABELS[m] if not det else None)
        lo, hi = x["delta"].min() - 0.15 * span, x["delta"].max() + 0.15 * span
        ax.plot([lo, 0, 0, hi], [1, 1, 0, 0], color=MUTED, lw=1.1, ls="--",
                label="Rational benchmark")
        ax.set_xlim(lo, hi)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel(r"Deviation gain $\Delta$ (payoff units)", color=INK)
        ax.grid(axis="y", color=GRID, lw=.7); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Obedience rate")
    axes[0].set_ylim(-.05, 1.05)
    from matplotlib.lines import Line2D
    h, l = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, mew=1.4, color=MUTED))
    l.append(r"Deterministic device ($\lambda=1$)")
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=7.5,
               bbox_to_anchor=(.5, -.01), handletextpad=.3, columnspacing=1.1)
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--results", help="results folder (default: results/)")
    args = ap.parse_args()
    res = Path(args.results) if args.results else SET.RESULTS_DIR
    pre = "pilot_" if args.pilot else ""
    out = res / ("analysis_pilot" if args.pilot else "analysis"); out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    warnings.filterwarnings("ignore")

    rb = random_baseline()
    rb.to_csv(out / "random_device_baseline.csv", index=False)
    print("Random-device baseline (Dirichlet(1) devices):\n", rb.round(4).to_string(index=False), "\n")

    a = load(res / f"{pre}partA.jsonl")
    if not a.empty:
        sc = score_part_a(a)
        sc.to_csv(out / "partA_scored.csv", index=False)
        for name, t in part_a_tables(sc).items():
            t.to_csv(out / f"partA_{name}.csv")
            print(f"Part A — {name}:\n{t}\n")

    b = load(res / f"{pre}partB.jsonl")
    if not b.empty:
        d = score_part_b(b)
        print(f"Part B parse rate: {b['ok'].mean():.3f} over {len(b)} calls\n")
        tab = obedience_table(d)
        tab.to_csv(out / "partB_obedience_by_cell.csv", index=False)
        h2 = h2_tables(d)
        for name, t in h2.items():
            t.to_csv(out / f"partB_H2_{name}.csv", index=False)
            print(f"Part B — H2 {name}:\n{t.round(3).to_string(index=False)}\n")
        rs = reasons_table(d)
        rs.to_csv(out / "partB_nonrational_reasons.csv", index=False)
        print(f"Part B — {len(rs)} decisions disagree with the rational benchmark; "
              f"stated reasons in partB_nonrational_reasons.csv\n")
        if (d["condition"] == "public").any():
            h3 = h3_tables(d)
            for name, t in h3.items():
                t.to_csv(out / f"partB_H3_{name}.csv", index=False)
                print(f"Part B — H3 {name}:\n{t.round(4).to_string(index=False)}\n")
        plot_obedience(h2["by_level"], out / "obedience_by_delta.png",
                       " (pilot, 1 sample per cell)" if args.pilot else "")
        print(f"Figures and tables written to {out}")


if __name__ == "__main__":
    main()
