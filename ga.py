"""Step 9b: Genetic algorithm that evolves the 10 rule weights.

  chromosome : 10 floats in [0, 1], one weight per fuzzy rule
  fitness    : mean per-user Spearman on the TRAIN split (earliest 70% of each user's ratings)
  operators  : tournament selection, blend crossover, gaussian mutation, elitism
  report     : the evolved weights are judged on the held-out TEST split (latest 30%)

Usage (from the project folder):
  python ga.py                      run all three membership families, then print the report
  python ga.py --family Gaussian    run one family only
  python ga.py --report             re-print the report and redraw the plot from saved results
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from deap import base, creator, tools

from evaluation import load_data, Evaluator
from fuzzy_system import (fuzzy_score, fuzzify, BASELINE_WEIGHTS, FAMILIES, RULES,
                          OUTPUT_SETS, Y)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "processed"
N_RULES = len(RULES)
CXPB, MUTPB, INDPB, SIGMA, TOURN, ELITE = 0.7, 0.4, 0.3, 0.15, 3, 2
OUT_RULES = {s: [i for i, (_, o) in enumerate(RULES) if o == s] for s in OUTPUT_SETS}


class FastScorer:
    """Same result as fuzzy_score, but the weight-independent part (rule firing strengths)
    is computed once, so each GA evaluation only redoes the cheap weighted part."""

    def __init__(self, items, family):
        mem = fuzzify(items, family)
        self.S = np.column_stack([np.minimum.reduce([mem[f][s] for f, s in conds]) for conds, _ in RULES])
        fam = FAMILIES[family]
        self.out_mf = {s: fam(Y, abc) for s, abc in OUTPUT_SETS.items()}

    def __call__(self, w):
        ws = self.S * np.asarray(w, float)
        mu = np.maximum.reduce([np.minimum(ws[:, idx].max(axis=1)[:, None], self.out_mf[s][None, :])
                                for s, idx in OUT_RULES.items()])
        den = mu.sum(axis=1)
        fired = den > 1e-9
        return np.where(fired, (mu * Y).sum(axis=1) / np.where(fired, den, 1), 0.5)


def clip01(ind):
    ind[:] = [min(1.0, max(0.0, x)) for x in ind]


def run_ga(scorer, ev_train, seed, pop_size, ngen):
    random.seed(seed)
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    tb = base.Toolbox()
    tb.register("select", tools.selTournament, tournsize=TOURN)
    tb.register("mate", tools.cxBlend, alpha=0.3)
    tb.register("mutate", tools.mutGaussian, mu=0.0, sigma=SIGMA, indpb=INDPB)

    def fit(ind):
        f = ev_train.evaluate(scorer(ind))["spearman"]
        return f if np.isfinite(f) else -1.0

    pop = [creator.Individual(random.random() for _ in range(N_RULES)) for _ in range(pop_size)]
    for ind in pop:
        ind.fitness.values = (fit(ind),)
    history = []

    def record(gen):
        fits = [i.fitness.values[0] for i in pop]
        history.append({"gen": gen, "best": max(fits), "mean": float(np.mean(fits))})

    record(0)
    for gen in range(1, ngen + 1):
        elite = [tb.clone(i) for i in tools.selBest(pop, ELITE)]
        off = [tb.clone(i) for i in tb.select(pop, len(pop) - ELITE)]
        for a, b in zip(off[::2], off[1::2]):
            if random.random() < CXPB:
                tb.mate(a, b); clip01(a); clip01(b)
                del a.fitness.values, b.fitness.values
        for m in off:
            if random.random() < MUTPB:
                tb.mutate(m); clip01(m)
                del m.fitness.values
        for i in off:
            if not i.fitness.valid:
                i.fitness.values = (fit(i),)
        pop = elite + off
        record(gen)
    best = tools.selBest(pop, 1)[0]
    return list(best), history


def run_family(family, seeds, pop_size, ngen):
    items, ratings, _ = load_data()
    train, test = Evaluator(ratings, "train"), Evaluator(ratings, "test")
    scorer = FastScorer(items, family)
    # correctness check: the fast path must reproduce fuzzy_score exactly
    w_check = np.random.default_rng(1).random(N_RULES)
    assert np.allclose(scorer(w_check), fuzzy_score(items, w_check, family)), "fast scorer mismatch"

    def judge(w):
        sc = fuzzy_score(items, w, family)          # official path, not the fast one
        a, b = train.evaluate(sc), test.evaluate(sc)
        return {"train_spearman": a["spearman"], "train_p": a["p_at_k"],
                "test_spearman": b["spearman"], "test_p": b["p_at_k"]}

    result = {"family": family, "pop": pop_size, "gens": ngen,
              "baseline_weights": [float(x) for x in BASELINE_WEIGHTS],
              "baseline": judge(BASELINE_WEIGHTS), "runs": []}
    for seed in seeds:
        t = time.time()
        w, hist = run_ga(scorer, train, seed, pop_size, ngen)
        res = judge(w)
        result["runs"].append({"seed": seed, "weights": [float(x) for x in w], "history": hist, **res})
        print(f"  {family:11s} seed {seed}: train {res['train_spearman']:.4f} "
              f"test {res['test_spearman']:.4f}  ({time.time() - t:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"ga_{family}.json").write_text(json.dumps(result, indent=1))
    return result


def report():
    items, ratings, _ = load_data()
    test = Evaluator(ratings, "test")
    ref = test.evaluate(items["rating_raw"].to_numpy())
    results = {f: json.loads((OUT / f"ga_{f}.json").read_text()) for f in FAMILIES if (OUT / f"ga_{f}.json").exists()}
    if not results:
        print("no saved results; run ga.py first"); return
    print(f"\nHeld-out TEST split. Reference, IMDb rating alone: Spearman {ref['spearman']:.4f}, P@10 {ref['p_at_k']:.4f}\n")
    hdr = f"{'family':12s}{'baseline Sp':>12s}{'GA Sp (mean±sd)':>20s}{'GA wins':>9s}{'baseline P@10':>15s}{'GA P@10 (mean±sd)':>22s}{'train Sp: base→GA':>22s}"
    print(hdr); print("-" * len(hdr))
    for fam, r in results.items():
        runs, b = r["runs"], r["baseline"]
        sp = np.array([x["test_spearman"] for x in runs]); pp = np.array([x["test_p"] for x in runs])
        tr = np.array([x["train_spearman"] for x in runs])
        wins = int((sp > b["test_spearman"]).sum())
        print(f"{fam:12s}{b['test_spearman']:12.4f}{sp.mean():12.4f} ± {sp.std():.4f}{wins:>5d}/{len(runs)}"
              f"{b['test_p']:15.4f}{pp.mean():14.4f} ± {pp.std():.4f}{b['train_spearman']:>12.4f} → {tr.mean():.4f}")
    print("\nLearned weights (mean over seeds) vs hand-tuned baseline:")
    rule_names = ["r1 High&PopHigh→High", "r2 High&New→High", "r3 High&PopLow→Med", "r4 Med&PopHigh→Med",
                  "r5 Med&New→Med", "r6 RatLow→Low", "r7 PopLow&Old→Low", "r8 Med&PopLow→Low",
                  "r9 High&PopMed→High", "r10 Med&PopMed→Med"]
    first = next(iter(results.values()))
    print(f"{'rule':24s}{'baseline':>9s}" + "".join(f"{f:>13s}" for f in results))
    for i, name in enumerate(rule_names):
        row = "".join(f"{np.mean([x['weights'][i] for x in r['runs']]):>8.2f}±{np.std([x['weights'][i] for x in r['runs']]):.2f}" for r in results.values())
        print(f"{name:24s}{first['baseline_weights'][i]:>9.2f}{row}")

    # convergence plot: one panel per family, mean over seeds
    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
    c_best, c_mean, c_base = "#2a78d6", "#eb6834", "#8a8984"
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), sharey=True, facecolor=surface)
    axes = np.atleast_1d(axes)
    for ax, (fam, r) in zip(axes, results.items()):
        gens = [h["gen"] for h in r["runs"][0]["history"]]
        best = np.mean([[h["best"] for h in x["history"]] for x in r["runs"]], axis=0)
        mean = np.mean([[h["mean"] for h in x["history"]] for x in r["runs"]], axis=0)
        ax.set_facecolor(surface)
        ax.axhline(r["baseline"]["train_spearman"], color=c_base, lw=1.5, ls="--", label="hand-tuned baseline")
        ax.plot(gens, mean, color=c_mean, lw=2, label="population mean")
        ax.plot(gens, best, color=c_best, lw=2, label="best in population")
        ax.set_title(fam, color=ink, fontsize=11)
        ax.set_xlabel("generation", color=ink2)
        ax.grid(color=grid, lw=0.8); ax.set_axisbelow(True)
        ax.tick_params(colors=ink2)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        for sp_ in ("left", "bottom"):
            ax.spines[sp_].set_color(grid)
    axes[0].set_ylabel("train Spearman (mean of seeds)", color=ink2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, labelcolor=ink2, fontsize=9, loc="lower center", ncol=3)
    fig.suptitle("GA convergence: fitness per generation", color=ink, fontsize=12)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig.savefig(OUT / "ga_convergence.png", dpi=120, facecolor=surface)
    print(f"\nsaved {OUT / 'ga_convergence.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="all", choices=["all", *FAMILIES])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--report", action="store_true", help="only print the report from saved results")
    a = ap.parse_args()
    if not a.report:
        for fam in (FAMILIES if a.family == "all" else [a.family]):
            run_family(fam, list(range(a.seeds)), a.pop, a.gens)
    report()
