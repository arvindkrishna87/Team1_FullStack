"""Reproduce the P3 Mathlete figures and all paired trial records (CPU only).
Run with OPENBLAS_NUM_THREADS=1 python tpe_gp.py OUTPUT_DIRECTORY.
Add --check-cohort on Quest to verify the eligible case counts.
Versions: numpy 2.3.4, scipy 1.16.3, optuna 4.7.0, matplotlib 3.10.7.
"""
import csv
import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.distributions import IntDistribution
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import ndtr, erfcx
from scipy.stats import binomtest

BASE = Path("/projects/e32998/STAT390_Fall2026/Team6_InstructionalTeam/Code/Presentation3_base_code")
PATCHES = Path("/projects/e32998/patch_images")

GRID = np.arange(101) / 100
START_COUNT, BUDGET, REPEATS = 5, 12, 100
EPSILON = 0.01
LOCAL = (GRID >= 0.10) & (GRID <= 0.30)
LENGTH, PRIOR_MEAN, PRIOR_SD = 0.15, 0.30, 0.25
COV = PRIOR_SD**2 * np.exp(-0.5 * ((GRID[:, None] - GRID) / LENGTH)**2)


def cohort():
    pattern = re.compile(r"case_(\d+)_.*\.png$", re.IGNORECASE)
    counts = Counter()
    for path in PATCHES.iterdir():
        match = pattern.fullmatch(path.name)
        if match:
            counts[int(match[1])] += 1
    with (BASE / "extra_benign_case_grade_match.csv").open() as stream:
        labels = {int(row["Case"]): float(row["Class"]) for row in csv.DictReader(stream)}
    groups = Counter(labels[i] for i in counts if labels[i] in {1, 3, 4})
    return len(counts), sum(counts.values()), groups[1], groups[3] + groups[4]


def loss(x):
    return np.minimum(0.20 + (x - 0.20)**2, 12 * (x - 0.80)**2)


def gp_choice(indices, objective=loss):
    """Fixed RBF prior; exact Gaussian EI on the shared finite search domain."""
    cross = COV[indices, :]
    gram = cross[:, indices] + 1e-10 * np.eye(len(indices))
    factor = cho_factor(gram, lower=True)
    y = objective(GRID[indices])
    mu = PRIOR_MEAN + cross.T @ cho_solve(factor, y - PRIOR_MEAN)
    projected = solve_triangular(factor[0], cross, lower=True)
    sigma = np.sqrt(np.maximum(0, PRIOR_SD**2 - (projected**2).sum(axis=0)))
    gain = y.min() - mu
    z = np.divide(gain, sigma, out=np.zeros_like(gain), where=sigma > 0)
    # Evaluate log EI so tiny positive improvements retain their ordering.
    score = np.full_like(gain, -np.inf)
    positive = sigma > 0
    central = positive & (z >= -5)
    score[central] = np.log(gain[central] * ndtr(z[central]) +
                            sigma[central] * np.exp(-z[central]**2/2) / np.sqrt(2*np.pi))
    tail = positive & (z < -5)
    a = -z[tail]
    # EI = sigma * phi(a) * [1 - a * Phi(-a) / phi(a)].
    factor_tail = 1 - a * np.sqrt(np.pi/2) * erfcx(a / np.sqrt(2))
    score[tail] = np.log(sigma[tail]) - a*a/2 - np.log(2*np.pi)/2 + np.log(factor_tail)
    score[indices] = -np.inf
    return int(np.argmax(score))


def verify_counterexample():
    """Place a narrow well at an unvisited grid point, then verify every GP choice."""
    def gp_path(objective):
        indices = [10, 15, 20, 25, 30]
        for _ in range(BUDGET):
            indices.append(gp_choice(indices, objective))
        return indices

    flat_path = gp_path(np.ones_like)
    target = next(i for i in range(75, 96) if i not in flat_path)

    def hidden_well(x):
        return 1 - np.maximum(0, 1 - np.abs(x - GRID[target]) / 0.005)

    assert gp_path(hidden_well) == flat_path
    assert hidden_well(GRID).min() == 0 and hidden_well(GRID[flat_path]).min() == 1
    print("Hidden optimum", GRID[target], "GP-EI adaptive indices", flat_path[5:])


def paired_run(seed):
    rng = np.random.default_rng(seed)
    start = rng.choice(np.flatnonzero(LOCAL), size=START_COUNT, replace=False).tolist()
    gp = start.copy()
    tpe = start.copy()
    sampler = optuna.samplers.TPESampler(seed=10000 + seed, n_startup_trials=0, n_ei_candidates=24)
    study = optuna.create_study(sampler=sampler, direction="minimize")
    distribution = IntDistribution(0, 100)
    for i in start:
        study.add_trial(optuna.trial.create_trial(params={"index": i}, distributions={"index": distribution},
                                                  value=float(loss(GRID[i]))))
    for _ in range(BUDGET):
        gp.append(gp_choice(gp))
        trial = study.ask()
        i = trial.suggest_int("index", 0, 100)
        study.tell(trial, float(loss(GRID[i])))
        tpe.append(i)
    return np.array([gp, tpe])


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    traces = np.array([paired_run(seed) for seed in range(REPEATS)])
    assert np.array_equal(traces[:, 0, :START_COUNT], traces[:, 1, :START_COUNT])
    names = ["GP-EI", "TPE"]
    with (out / "trials.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "method", "evaluation", "phase", "x", "loss"])
        for seed, pair in enumerate(traces):
            for name, indices in zip(names, pair):
                for step, i in enumerate(indices):
                    writer.writerow([seed, name, step+1, "start" if step < START_COUNT else "adaptive",
                                     GRID[i], float(loss(GRID[i]))])
    adaptive = traces[:, :, START_COUNT:]
    hits = loss(GRID[adaptive]).min(axis=2) <= EPSILON
    local = LOCAL[adaptive].mean(axis=2)
    for k, name in enumerate(names):
        count = int(hits[:, k].sum())
        interval = binomtest(count, REPEATS).proportion_ci()
        print(name, "hits", count, "95% exact interval", interval, "mean local fraction", local[:, k].mean())
    print("paired outcomes both, GP only, TPE only, neither", [
        int(((hits[:, 0] == a) & (hits[:, 1] == b)).sum()) for a, b in [(1,1),(1,0),(0,1),(0,0)]])
    print("seed zero indices", traces[0].tolist())
    fig, (ax, path) = plt.subplots(2, 1, figsize=(7.0, 3.4), sharex=True,
                                   gridspec_kw={"height_ratios":[1.5,1]}, layout="constrained")
    ax.plot(GRID, loss(GRID), color="#032C3C", linewidth=2)
    ax.scatter(GRID[traces[0,0,:START_COUNT]], loss(GRID[traces[0,0,:START_COUNT]]),
               color="#032C3C", s=25, label="Shared start", zorder=3)
    ax.legend(frameon=False, loc="upper right"); ax.set_ylabel("Toy loss")
    for a in (ax,path):
        a.axvspan(.1,.3,color="#D3C3B2",alpha=.45)
        a.axvspan(.775,.825,color="#A7291E",alpha=.14)
        a.spines[["top","right"]].set_visible(False)
    for name, series, marker, color in zip(names, adaptive[0], ["o","x"], ["#032C3C","#A7291E"]):
        path.scatter(GRID[series], np.arange(1,BUDGET+1), marker=marker, s=22, color=color, label=name)
    path.set(yticks=[1,6,12],ylabel="Trial",xlabel="Candidate setting x", xlim=(-.015,1.015))
    path.legend(loc="upper left", bbox_to_anchor=(1,1), frameon=False)
    fig.savefig(out / "search_paths.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--check-cohort", action="store_true", help="Check the local Quest patch data.")
    args = parser.parse_args()
    if args.check_cohort:
        print("Raw cases, raw patches, eligible benign, eligible high-grade", cohort())
    verify_counterexample()
    main(args.output_directory)
