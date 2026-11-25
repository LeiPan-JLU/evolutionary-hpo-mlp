# -*- coding: utf-8 -*-
"""统计检验与论文图：收敛曲线、箱线图、DE 与 jDE 的消融对比。"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

ALGOS = ["RandomSearch", "RealGA", "LDIW-PSO", "DE", "jDE"]
LABELS = {
    "RandomSearch": "Random Search",
    "RealGA": "Real-coded GA",
    "LDIW-PSO": "LDIW-PSO",
    "DE": "DE/rand/1/bin",
    "jDE": "jDE",
}
COLORS = {
    "RandomSearch": "#7f7f7f",
    "RealGA": "#1f77b4",
    "LDIW-PSO": "#ff7f0e",
    "DE": "#2ca02c",
    "jDE": "#d62728",
}
STYLES = {
    "RandomSearch": (":", "o"),
    "RealGA": ("-", "s"),
    "LDIW-PSO": ("-.", "^"),
    "DE": ("--", "D"),
    "jDE": ("-", "*"),
}
MAX_FES = 400
POP_SIZE = 20

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120


def load_results():
    with open(os.path.join("results", "raw_results.csv"), encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    data = {algo: [] for algo in ALGOS}
    times = {algo: [] for algo in ALGOS}
    hits = {algo: [] for algo in ALGOS}
    for row in rows:
        data[row["algorithm"]].append(float(row["best_error"]))
        times[row["algorithm"]].append(float(row["time_sec"]))
        hits[row["algorithm"]].append(int(row["cache_hits"]))
    return ({k: np.array(v) for k, v in data.items()},
            {k: np.array(v) for k, v in times.items()},
            {k: np.array(v) for k, v in hits.items()})


def make_statistics(data, times, hits):
    ref = data["jDE"]
    rows = []
    for algo in ALGOS:
        vals = data[algo]
        if algo == "jDE":
            p_value, mark = float("nan"), "—"
        else:
            diff = vals - ref
            if np.allclose(diff, 0.0):
                p_value = 1.0
            else:
                p_value = float(wilcoxon(vals, ref).pvalue)
            if p_value < 0.05 and vals.mean() < ref.mean():
                mark = "+"
            elif p_value < 0.05 and vals.mean() > ref.mean():
                mark = "-"
            else:
                mark = "≈"
        rows.append({
            "algorithm": algo,
            "best": float(vals.min()),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)),
            "median": float(np.median(vals)),
            "worst": float(vals.max()),
            "mean_time": float(times[algo].mean()),
            "std_time": float(times[algo].std(ddof=1)),
            "mean_cache_hits": float(hits[algo].mean()),
            "p_value": p_value,
            "mark": mark,
        })
    with open(os.path.join("results", "statistics.csv"), "w", newline="", encoding="utf-8") as handle:
        fields = ["algorithm", "best", "mean", "std", "median", "worst",
                  "mean_time", "std_time", "mean_cache_hits", "p_value", "mark"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.3)


def plot_convergence(history):
    fes = np.arange(POP_SIZE, MAX_FES + 1, POP_SIZE)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for algo in ALGOS:
        mean = history[ALGOS.index(algo)].mean(axis=0)
        std = history[ALGOS.index(algo)].std(axis=0, ddof=1)
        linestyle, marker = STYLES[algo]
        ax.plot(fes, mean, linestyle=linestyle, marker=marker, markevery=2,
                color=COLORS[algo], label=LABELS[algo], linewidth=1.6, markersize=4)
        ax.fill_between(fes, mean - std, mean + std, color=COLORS[algo], alpha=0.13, linewidth=0)
    ax.set_xlabel("函数评价次数 (FEs)")
    ax.set_ylabel("5 折交叉验证错误率")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "convergence.png"), dpi=300)
    plt.close(fig)


def plot_boxplot(data):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    rng = np.random.default_rng(7)
    for i, algo in enumerate(ALGOS, start=1):
        vals = data[algo]
        bp = ax.boxplot(vals, positions=[i], widths=0.5, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS[algo])
            patch.set_alpha(0.35)
        for median in bp["medians"]:
            median.set_color(COLORS[algo])
            median.set_linewidth(1.5)
        x = i + rng.uniform(-0.12, 0.12, size=vals.size)
        ax.scatter(x, vals, s=12, color=COLORS[algo], alpha=0.7, edgecolors="none", zorder=3)
    ax.set_xticks(range(1, len(ALGOS) + 1))
    ax.set_xticklabels([LABELS[a] for a in ALGOS], fontsize=9)
    ax.set_ylabel("最终验证错误率")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "boxplot.png"), dpi=300)
    plt.close(fig)


def plot_ablation(history, jde_params):
    fes = np.arange(POP_SIZE, MAX_FES + 1, POP_SIZE)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))

    ax = axes[0]
    for algo in ["DE", "jDE"]:
        mean = history[ALGOS.index(algo)].mean(axis=0)
        std = history[ALGOS.index(algo)].std(axis=0, ddof=1)
        linestyle, marker = STYLES[algo]
        ax.plot(fes, mean, linestyle=linestyle, marker=marker, markevery=2,
                color=COLORS[algo], label=LABELS[algo], linewidth=1.6, markersize=4)
        ax.fill_between(fes, mean - std, mean + std, color=COLORS[algo], alpha=0.15, linewidth=0)
    ax.set_xlabel("函数评价次数 (FEs)")
    ax.set_ylabel("5 折交叉验证错误率")
    ax.set_title("(a) 收敛曲线对比", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    style_axis(ax)

    ax = axes[1]
    gens = np.arange(1, jde_params.shape[1] + 1)
    mean_f = jde_params[:, :, 0].mean(axis=0)
    mean_cr = jde_params[:, :, 1].mean(axis=0)
    ax.plot(gens, mean_f, color="#1f77b4", marker="o", markersize=3, label="平均 F")
    ax.plot(gens, mean_cr, color="#d62728", marker="s", markersize=3, label="平均 CR")
    ax.set_xlabel("进化代数")
    ax.set_ylabel("参数均值")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("(b) jDE 参数均值演化", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    style_axis(ax)

    fig.tight_layout()
    fig.savefig(os.path.join("figures", "ablation.png"), dpi=300)
    plt.close(fig)


def main():
    os.makedirs("figures", exist_ok=True)
    data, times, hits = load_results()
    rows = make_statistics(data, times, hits)
    print(f"{'algorithm':14s} {'mean':>8s} {'std':>8s} {'median':>8s} {'time':>8s} {'hits':>6s} {'p':>8s} {'mark':>4s}")
    for row in rows:
        print(f"{row['algorithm']:14s} {row['mean']:8.4f} {row['std']:8.4f} "
              f"{row['median']:8.4f} {row['mean_time']:8.1f} {row['mean_cache_hits']:6.1f} "
              f"{row['p_value']:8.4f} {row['mark']:>4s}")

    history = np.load(os.path.join("results", "convergence_history.npy"))
    jde_params = np.load(os.path.join("results", "jde_params_history.npy"))
    plot_convergence(history)
    plot_boxplot(data)
    plot_ablation(history, jde_params)
    print("saved figures: convergence.png, boxplot.png, ablation.png")


if __name__ == "__main__":
    main()
