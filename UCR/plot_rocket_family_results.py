import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns


METHOD_LABELS = {
    "kfstune": "KFS-TUNE",
    "rocket": "ROCKET",
    "minirocket": "MiniROCKET",
    "multirocket": "MultiROCKET",
    "srocket": "S-ROCKET",
    "pocket": "POCKET",
    "detach_rocket": "Detach-ROCKET",
}

METHOD_ORDER = [
    "kfstune",
    "rocket",
    "minirocket",
    "multirocket",
    "srocket",
    "pocket",
    "detach_rocket",
]

PALETTE = {
    "KFS-TUNE": "#2f6f73",
    "ROCKET": "#b44746",
    "MiniROCKET": "#4b77be",
    "MultiROCKET": "#7a5195",
    "S-ROCKET": "#d28b26",
    "POCKET": "#5f8d4e",
    "Detach-ROCKET": "#6b6f77",
}

HEATMAP_CMAP = LinearSegmentedColormap.from_list("white_to_teal", ["#ffffff", "#2f6f73"])


def seconds_to_label(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}m"


def load_results(path):
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    df["method_label"] = pd.Categorical(
        df["method"].map(METHOD_LABELS),
        categories=[METHOD_LABELS[m] for m in METHOD_ORDER],
        ordered=True,
    )
    df["total_time_s"] = pd.to_numeric(df["total_time_s"], errors="coerce")
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    return df


def style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelweight": "semibold",
            "font.family": "DejaVu Sans",
        }
    )


def save(fig, output_dir, name):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_box(df, output_dir):
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    order = [METHOD_LABELS[m] for m in METHOD_ORDER]
    sns.violinplot(
        data=df,
        x="method_label",
        y="accuracy",
        order=order,
        palette=PALETTE,
        inner=None,
        cut=0,
        linewidth=0,
        alpha=0.25,
        ax=ax,
    )
    sns.boxplot(
        data=df,
        x="method_label",
        y="accuracy",
        order=order,
        width=0.28,
        showcaps=True,
        boxprops={"facecolor": "white", "edgecolor": "#34383c", "linewidth": 1.1},
        whiskerprops={"color": "#34383c", "linewidth": 1.1},
        medianprops={"color": "#111111", "linewidth": 1.6},
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="method_label",
        y="accuracy",
        order=order,
        color="#252525",
        size=2.2,
        alpha=0.28,
        jitter=0.22,
        ax=ax,
    )
    means = df.groupby("method_label", observed=True)["accuracy"].mean().reindex(order)
    for i, mean in enumerate(means):
        ax.scatter(i, mean, marker="D", s=36, color="#f2c14e", edgecolor="#222", linewidth=0.7, zorder=5)
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.03)
    ax.tick_params(axis="x", rotation=22)
    save(fig, output_dir, "accuracy_distribution")


def plot_time_box(df, output_dir):
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    order = [METHOD_LABELS[m] for m in METHOD_ORDER]
    sns.boxplot(
        data=df,
        x="method_label",
        y="total_time_s",
        order=order,
        palette=PALETTE,
        showfliers=False,
        width=0.56,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="method_label",
        y="total_time_s",
        order=order,
        color="#222222",
        size=2.1,
        alpha=0.28,
        jitter=0.20,
        ax=ax,
    )
    medians = df.groupby("method_label", observed=True)["total_time_s"].median().reindex(order)
    for i, median in enumerate(medians):
        ax.text(i, median * 1.15, seconds_to_label(median), ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("End-to-end runtime, seconds (log scale)")
    ax.tick_params(axis="x", rotation=22)
    save(fig, output_dir, "runtime_distribution_log")


def plot_accuracy_time_scatter(df, output_dir):
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    summary = (
        df.groupby(["method", "method_label"], observed=True)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            median_time_s=("total_time_s", "median"),
            mean_time_s=("total_time_s", "mean"),
            mean_features=("retained_features", "mean"),
        )
        .reset_index()
    )
    summary["method_label"] = summary["method"].map(METHOD_LABELS)
    summary = summary[summary["method"].isin(METHOD_ORDER)]
    max_features = summary["mean_features"].max()
    sizes = 180 + 720 * np.sqrt(summary["mean_features"].fillna(max_features) / max_features)
    for _, row in summary.iterrows():
        ax.scatter(
            row["median_time_s"],
            row["mean_accuracy"],
            s=float(sizes.loc[row.name]) if row.name in sizes.index else 420,
            color=PALETTE[row["method_label"]],
            edgecolor="#202020",
            linewidth=0.8,
            alpha=0.86,
        )
        ax.annotate(
            row["method_label"],
            (row["median_time_s"], row["mean_accuracy"]),
            xytext=(7, 4),
            textcoords="offset points",
            fontsize=9,
            weight="semibold",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Median end-to-end runtime, seconds (log scale)")
    ax.set_ylabel("Mean accuracy")
    ax.set_ylim(max(0.72, summary["mean_accuracy"].min() - 0.02), min(0.90, summary["mean_accuracy"].max() + 0.02))
    ax.grid(True, which="both", axis="x", alpha=0.18)
    save(fig, output_dir, "accuracy_speed_tradeoff")


def plot_average_rank(df, output_dir):
    pivot = df.pivot_table(index="dataset", columns="method", values="accuracy", aggfunc="mean")
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    rank_summary = (
        ranks.mean()
        .reindex(METHOD_ORDER)
        .rename("average_rank")
        .reset_index()
        .dropna()
    )
    rank_summary["method_label"] = rank_summary["method"].map(METHOD_LABELS)
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    sns.barplot(
        data=rank_summary,
        y="method_label",
        x="average_rank",
        order=[METHOD_LABELS[m] for m in METHOD_ORDER],
        palette=PALETTE,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=4, fontsize=9)
    ax.invert_xaxis()
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_ylabel("")
    save(fig, output_dir, "average_rank")


def plot_wins_vs_kfstune(df, output_dir):
    pivot = df.pivot_table(index="dataset", columns="method", values="accuracy", aggfunc="mean")
    rows = []
    for method in METHOD_ORDER:
        if method == "kfstune" or method not in pivot or "kfstune" not in pivot:
            continue
        both = pivot[["kfstune", method]].dropna()
        diff = both[method] - both["kfstune"]
        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "Baseline better": int((diff > 1e-12).sum()),
                "Tie": int((diff.abs() <= 1e-12).sum()),
                "KFS-TUNE better": int((diff < -1e-12).sum()),
            }
        )
    wtl = pd.DataFrame(rows)
    colors = {"Baseline better": "#b85b5b", "Tie": "#b7b7b7", "KFS-TUNE better": "#4f8f6f"}
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    left = np.zeros(len(wtl))
    y = np.arange(len(wtl))
    for outcome in ["Baseline better", "Tie", "KFS-TUNE better"]:
        values = wtl[outcome].to_numpy()
        ax.barh(y, values, left=left, color=colors[outcome], edgecolor="white", label=outcome)
        for i, value in enumerate(values):
            if value > 0:
                ax.text(left[i] + value / 2, i, str(value), ha="center", va="center", color="white", fontsize=9, weight="bold")
        left += values
    ax.set_yticks(y, wtl["method_label"])
    ax.set_xlabel("Datasets")
    ax.legend(loc="lower right", frameon=True)
    save(fig, output_dir, "win_tie_loss_vs_kfstune")


def plot_accuracy_delta_vs_kfstune(df, output_dir):
    pivot = df.pivot_table(index="dataset", columns="method", values="accuracy", aggfunc="mean")
    if "kfstune" not in pivot:
        return
    rows = []
    for method in METHOD_ORDER:
        if method == "kfstune" or method not in pivot:
            continue
        diff = (pivot[method] - pivot["kfstune"]).dropna()
        for dataset, value in diff.items():
            rows.append({"dataset": dataset, "method": method, "method_label": METHOD_LABELS[method], "accuracy_delta": value})
    delta = pd.DataFrame(rows)
    order = [METHOD_LABELS[m] for m in METHOD_ORDER if m != "kfstune"]
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    sns.boxplot(
        data=delta,
        x="method_label",
        y="accuracy_delta",
        order=order,
        palette={k: PALETTE[k] for k in order},
        showfliers=False,
        width=0.52,
        ax=ax,
    )
    sns.stripplot(
        data=delta,
        x="method_label",
        y="accuracy_delta",
        order=order,
        color="#222222",
        size=2.0,
        alpha=0.25,
        jitter=0.20,
        ax=ax,
    )
    ax.axhline(0, color="#111111", linewidth=1.2, linestyle="--")
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy difference vs KFS-TUNE")
    ax.tick_params(axis="x", rotation=22)
    save(fig, output_dir, "accuracy_delta_vs_kfstune")


def plot_runtime_ratio_vs_kfstune(df, output_dir):
    pivot = df.pivot_table(index="dataset", columns="method", values="total_time_s", aggfunc="mean")
    if "kfstune" not in pivot:
        return
    rows = []
    for method in METHOD_ORDER:
        if method == "kfstune" or method not in pivot:
            continue
        ratio = (pivot[method] / pivot["kfstune"]).replace([np.inf, -np.inf], np.nan).dropna()
        for dataset, value in ratio.items():
            rows.append({"dataset": dataset, "method": method, "method_label": METHOD_LABELS[method], "runtime_ratio": value})
    ratios = pd.DataFrame(rows)
    order = [METHOD_LABELS[m] for m in METHOD_ORDER if m != "kfstune"]
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    sns.boxplot(
        data=ratios,
        x="method_label",
        y="runtime_ratio",
        order=order,
        palette={k: PALETTE[k] for k in order},
        showfliers=False,
        width=0.52,
        ax=ax,
    )
    sns.stripplot(
        data=ratios,
        x="method_label",
        y="runtime_ratio",
        order=order,
        color="#222222",
        size=2.0,
        alpha=0.25,
        jitter=0.20,
        ax=ax,
    )
    ax.axhline(1, color="#111111", linewidth=1.2, linestyle="--")
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("Runtime ratio vs KFS-TUNE (log scale)")
    ax.tick_params(axis="x", rotation=22)
    save(fig, output_dir, "runtime_ratio_vs_kfstune")


def plot_utility_vs_kfstune(df, output_dir, penalty=0.02):
    working = df.copy()
    working["utility"] = working["accuracy"] - penalty * np.log10(np.maximum(working["total_time_s"], 1e-9))
    pivot = working.pivot_table(index="dataset", columns="method", values="utility", aggfunc="mean")
    if "kfstune" not in pivot:
        return
    rows = []
    for method in METHOD_ORDER:
        if method == "kfstune" or method not in pivot:
            continue
        diff = (pivot[method] - pivot["kfstune"]).dropna()
        for dataset, value in diff.items():
            rows.append({"dataset": dataset, "method": method, "method_label": METHOD_LABELS[method], "utility_delta": value})
    util = pd.DataFrame(rows)
    order = [METHOD_LABELS[m] for m in METHOD_ORDER if m != "kfstune"]
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    sns.boxplot(
        data=util,
        x="method_label",
        y="utility_delta",
        order=order,
        palette={k: PALETTE[k] for k in order},
        showfliers=False,
        width=0.52,
        ax=ax,
    )
    sns.stripplot(
        data=util,
        x="method_label",
        y="utility_delta",
        order=order,
        color="#222222",
        size=2.0,
        alpha=0.25,
        jitter=0.20,
        ax=ax,
    )
    ax.axhline(0, color="#111111", linewidth=1.2, linestyle="--")
    ax.set_xlabel("")
    ax.set_ylabel("Utility difference vs KFS-TUNE")
    ax.tick_params(axis="x", rotation=22)
    save(fig, output_dir, "utility_delta_vs_kfstune")


def plot_heatmap(df, output_dir, top_n):
    pivot = df.pivot_table(index="dataset", columns="method", values="accuracy", aggfunc="mean")
    complete = pivot.dropna()
    if len(complete) > top_n:
        spread = complete.max(axis=1) - complete.min(axis=1)
        complete = complete.loc[spread.sort_values(ascending=False).head(top_n).index]
    complete = complete[METHOD_ORDER]
    complete.columns = [METHOD_LABELS[c] for c in complete.columns]
    fig_height = max(7.0, 0.26 * len(complete))
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    sns.heatmap(
        complete,
        cmap=HEATMAP_CMAP,
        vmin=0,
        vmax=1,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Accuracy"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    save(fig, output_dir, "accuracy_heatmap_top_differences")


def write_plot_summary(df, output_dir):
    summary = (
        df.groupby(["method", "method_label"], observed=True)
        .agg(
            datasets=("dataset", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            median_accuracy=("accuracy", "median"),
            mean_time_s=("total_time_s", "mean"),
            median_time_s=("total_time_s", "median"),
            mean_retained_features=("retained_features", "mean"),
            mean_retained_kernels=("retained_kernels", "mean"),
            mean_memory_delta_mb=("memory_delta_mb", "mean"),
        )
        .reset_index()
    )
    summary["method"] = pd.Categorical(summary["method"], categories=METHOD_ORDER, ordered=True)
    summary = summary.sort_values("method")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "plot_summary_by_method.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Create publication-ready plots for ROCKET-family baseline results.")
    parser.add_argument(
        "--results-csv",
        default="results/rocket_family_baselines/per_dataset_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/rocket_family_baselines/plots",
    )
    parser.add_argument("--heatmap-top-n", type=int, default=45)
    return parser.parse_args()


def main():
    args = parse_args()
    style()
    output_dir = Path(args.output_dir)
    df = load_results(args.results_csv)
    if df.empty:
        raise SystemExit("No ok result rows found.")

    plot_accuracy_box(df, output_dir)
    plot_time_box(df, output_dir)
    plot_accuracy_time_scatter(df, output_dir)
    plot_average_rank(df, output_dir)
    plot_wins_vs_kfstune(df, output_dir)
    plot_accuracy_delta_vs_kfstune(df, output_dir)
    plot_runtime_ratio_vs_kfstune(df, output_dir)
    plot_utility_vs_kfstune(df, output_dir)
    plot_heatmap(df, output_dir, args.heatmap_top_n)
    write_plot_summary(df, output_dir)
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
