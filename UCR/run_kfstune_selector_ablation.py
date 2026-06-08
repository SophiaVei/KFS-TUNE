import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KFSTUNE_functions import apply_kernels, generate_kernels
from UCR.run_rocket_family_baselines import UCR_BAKEOFF_DATASETS, load_ucr_dataset


SELECTOR_ORDER = ["chi2", "anova_f", "mutual_info", "random"]
SELECTOR_LABELS = {
    "chi2": r"$\chi^2$",
    "anova_f": "ANOVA-F",
    "mutual_info": "Mutual information",
    "random": "Random",
}
PALETTE = {
    r"$\chi^2$": "#0f766e",
    "ANOVA-F": "#b45309",
    "Mutual information": "#4f46e5",
    "Random": "#52525b",
}
SUMMARY_CMAP = LinearSegmentedColormap.from_list("white_to_teal", ["#ffffff", "#0f766e"])


@dataclass
class SelectorResult:
    dataset: str
    selector: str
    accuracy: float | None = None
    train_transform_time_s: float | None = None
    feature_selection_time_s: float | None = None
    training_time_s: float | None = None
    test_transform_time_s: float | None = None
    inference_time_s: float | None = None
    total_time_s: float | None = None
    retained_features: int | None = None
    memory_delta_mb: float | None = None
    status: str = "ok"
    error: str = ""


def rss_mb():
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def fit_ridge(x_train, y_train):
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    clf.fit(x_train, y_train)
    return clf


def select_features(z_train, y_train, selector_name, num_features, seed):
    num_features = min(num_features, z_train.shape[1])
    if selector_name == "chi2":
        scores = chi2(z_train, y_train)[0]
    elif selector_name == "anova_f":
        scores = f_classif(z_train, y_train)[0]
    elif selector_name == "mutual_info":
        scores = mutual_info_classif(z_train, y_train, random_state=seed)
    elif selector_name == "random":
        rng = np.random.default_rng(seed)
        indices = rng.choice(z_train.shape[1], size=num_features, replace=False)
        indices.sort()
        return indices
    else:
        raise ValueError(f"Unknown selector: {selector_name}")

    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    indices = np.argpartition(scores, -num_features)[-num_features:]
    indices = indices[np.argsort(scores[indices])[::-1]]
    indices.sort()
    return indices


def run_selector(dataset, x_train, y_train, x_test, y_test, args):
    result = SelectorResult(dataset=dataset, selector=args.worker_selector)
    mem0 = rss_mb()
    start_total = time.perf_counter()
    try:
        np.random.seed(args.seed)
        avg_len = int(np.mean([len(x) for x in x_train]))
        kernels = generate_kernels(x_train.shape[1], args.num_kernels, avg_len)

        start = time.perf_counter()
        z_train = apply_kernels(x_train, kernels)
        scaler = MinMaxScaler()
        z_train = scaler.fit_transform(z_train)
        result.train_transform_time_s = time.perf_counter() - start

        start = time.perf_counter()
        selected = select_features(
            z_train,
            y_train,
            args.worker_selector,
            args.num_features,
            args.seed,
        )
        x_train_sel = z_train[:, selected]
        result.feature_selection_time_s = time.perf_counter() - start
        result.retained_features = int(len(selected))

        start = time.perf_counter()
        clf = fit_ridge(x_train_sel, y_train)
        result.training_time_s = time.perf_counter() - start

        start = time.perf_counter()
        z_test = apply_kernels(x_test, kernels)
        z_test = scaler.transform(z_test)
        x_test_sel = z_test[:, selected]
        result.test_transform_time_s = time.perf_counter() - start

        start = time.perf_counter()
        pred = clf.predict(x_test_sel)
        result.inference_time_s = time.perf_counter() - start
        result.accuracy = accuracy_score(y_test, pred)
        result.total_time_s = time.perf_counter() - start_total
    except Exception as exc:
        result.status = "error"
        result.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"

    mem1 = rss_mb()
    result.memory_delta_mb = None if mem0 is None or mem1 is None else mem1 - mem0
    return result


def append_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def existing_pairs(path, rerun_statuses):
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    if rerun_statuses:
        df = df[~df["status"].astype(str).isin(set(rerun_statuses))]
    return set(zip(df["dataset"].astype(str), df["selector"].astype(str)))


def remove_rerun_rows(path, datasets, selectors, statuses):
    if not path.exists() or not statuses:
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    drop = (
        df["dataset"].astype(str).isin(set(datasets))
        & df["selector"].astype(str).isin(set(selectors))
        & df["status"].astype(str).isin(set(statuses))
    )
    if drop.any():
        df.loc[~drop].to_csv(path, index=False)


def complete_selector_cases(df, required_selectors=None, dataset_filter=None):
    required_selectors = required_selectors or SELECTOR_ORDER
    ok = df[df["status"] == "ok"].copy()
    if dataset_filter is not None:
        ok = ok[ok["dataset"].astype(str).isin(set(map(str, dataset_filter)))].copy()
    completed = ok.groupby("dataset")["selector"].apply(lambda s: set(s.astype(str)))
    keep_datasets = [
        dataset
        for dataset, selectors in completed.items()
        if set(required_selectors).issubset(selectors)
    ]
    return ok[ok["dataset"].isin(keep_datasets)].copy()


def summarize_results(df, baseline_selector="chi2"):
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    pivot = ok.pivot_table(index="dataset", columns="selector", values="accuracy", aggfunc="mean")
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    rows = []
    for selector in SELECTOR_ORDER:
        if selector not in ok["selector"].unique():
            continue
        group = ok[ok["selector"] == selector]
        wins = ties = losses = np.nan
        if selector != baseline_selector and baseline_selector in pivot and selector in pivot:
            both = pivot[[baseline_selector, selector]].dropna()
            diff = both[selector] - both[baseline_selector]
            wins = int((diff > 1e-12).sum())
            ties = int((diff.abs() <= 1e-12).sum())
            losses = int((diff < -1e-12).sum())
        rows.append(
            {
                "selector": selector,
                "selector_label": SELECTOR_LABELS[selector],
                "datasets_completed": int(group["dataset"].nunique()),
                "mean_accuracy": group["accuracy"].mean(),
                "median_accuracy": group["accuracy"].median(),
                "average_rank": ranks[selector].mean() if selector in ranks else np.nan,
                "median_total_time_s": group["total_time_s"].median(),
                "median_feature_selection_time_s": group["feature_selection_time_s"].median(),
                "mean_retained_features": group["retained_features"].mean(),
                "wins_vs_chi2": wins,
                "ties_vs_chi2": ties,
                "losses_vs_chi2": losses,
                "mean_memory_delta_mb": group["memory_delta_mb"].mean(),
            }
        )
    summary = pd.DataFrame(rows)
    summary["selector"] = pd.Categorical(summary["selector"], categories=SELECTOR_ORDER, ordered=True)
    summary = summary.sort_values("selector")
    return summary


def write_summary(results_csv, summary_csv, baseline_selector="chi2", complete_only=False):
    df = pd.read_csv(results_csv)
    if complete_only:
        df = complete_selector_cases(df)
    summary = summarize_results(df, baseline_selector=baseline_selector)
    summary.to_csv(summary_csv, index=False)
    return summary


def style():
    sns.set_theme(style="ticks", context="paper", font_scale=1.08)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "figure.facecolor": "#fbfaf6",
            "axes.facecolor": "#fbfaf6",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#27272a",
            "axes.labelweight": "semibold",
            "font.family": "DejaVu Sans",
        }
    )


def save(fig, output_dir, name):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", bbox_inches="tight", transparent=True)
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight", transparent=True)
    plt.close(fig)


def selector_summary_for_plots(df):
    ok = df[df["status"] == "ok"].copy()
    ok["selector_label"] = ok["selector"].map(SELECTOR_LABELS)
    summary = (
        ok.groupby(["selector", "selector_label"], observed=True)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            median_accuracy=("accuracy", "median"),
            median_runtime=("total_time_s", "median"),
            median_selection_time=("feature_selection_time_s", "median"),
            accuracy_iqr=("accuracy", lambda s: s.quantile(0.75) - s.quantile(0.25)),
        )
        .reset_index()
    )
    summary["selector"] = pd.Categorical(summary["selector"], categories=SELECTOR_ORDER, ordered=True)
    return summary.sort_values("selector"), ok


def plot_selector_accuracy_ecdf(df, output_dir):
    _, ok = selector_summary_for_plots(df)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for selector in SELECTOR_ORDER:
        group = ok[ok["selector"] == selector]["accuracy"].dropna().sort_values()
        if group.empty:
            continue
        x = group.to_numpy()
        y = 1 - np.arange(1, len(x) + 1) / len(x)
        label = SELECTOR_LABELS[selector]
        ax.step(x, y, where="post", color=PALETTE[label], linewidth=2.2, label=label)
        ax.fill_between(x, y, step="post", alpha=0.06, color=PALETTE[label])
    ax.set_xlabel("Accuracy threshold")
    ax.set_ylabel("Fraction above threshold")
    ax.set_xlim(0.35, 1.0)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=2, loc="lower left")
    ax.grid(axis="y", color="#d4d4d8", linewidth=0.8)
    save(fig, output_dir, "selector_accuracy_ecdf")


def plot_selector_radar(df, summary, output_dir):
    if summary.empty:
        return
    working = summary.copy()
    rank_map = {}
    ok = df[df["status"] == "ok"].copy()
    pivot = ok.pivot_table(index="dataset", columns="selector", values="accuracy", aggfunc="mean")
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    for selector in SELECTOR_ORDER:
        if selector in ranks:
            rank_map[selector] = ranks[selector].mean()
    working["average_rank"] = working["selector"].astype(str).map(rank_map)

    metrics = [
        ("Accuracy", working["mean_accuracy"], True),
        ("Rank", working["average_rank"], False),
        ("Runtime", working["median_runtime"], False),
        ("Selection speed", working["median_selection_time"], False),
        ("Stability", working["accuracy_iqr"], False),
    ]

    labels = [m[0] for m in metrics]
    values_by_selector = {}
    for _, row in working.iterrows():
        vals = []
        for _, series, higher_is_better in metrics:
            lo = float(np.nanmin(series))
            hi = float(np.nanmax(series))
            val = row[series.name]
            if not np.isfinite(val) or hi == lo:
                score = 0.7
            else:
                score = (val - lo) / (hi - lo)
                if not higher_is_better:
                    score = 1 - score
            vals.append(float(np.clip(score, 0, 1)))
        values_by_selector[row["selector_label"]] = vals

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6.4, 6.4), subplot_kw={"projection": "polar"})
    ax.set_facecolor("#fbfaf6")
    for label, vals in values_by_selector.items():
        vals = vals + vals[:1]
        ax.plot(angles, vals, color=PALETTE[label], linewidth=2.0, label=label)
        ax.fill(angles, vals, color=PALETTE[label], alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.grid(color="#d4d4d8", linewidth=0.8)
    ax.spines["polar"].set_color("#a1a1aa")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.28, 1.12))
    save(fig, output_dir, "selector_tradeoff_radar")


def plot_selector_delta_raincloud(df, output_dir):
    ok = df[df["status"] == "ok"].copy()
    pivot = ok.pivot_table(index="dataset", columns="selector", values="accuracy", aggfunc="mean")
    if "chi2" not in pivot:
        return
    rows = []
    for selector in SELECTOR_ORDER:
        if selector == "chi2" or selector not in pivot:
            continue
        diff = (pivot[selector] - pivot["chi2"]).dropna()
        for dataset, value in diff.items():
            rows.append(
                {
                    "dataset": dataset,
                    "selector": selector,
                    "selector_label": SELECTOR_LABELS[selector],
                    "accuracy_delta": value,
                }
            )
    delta = pd.DataFrame(rows)
    if delta.empty:
        return
    order = [SELECTOR_LABELS[s] for s in SELECTOR_ORDER if s != "chi2" and s in pivot]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    sns.violinplot(
        data=delta,
        x="selector_label",
        y="accuracy_delta",
        hue="selector_label",
        order=order,
        hue_order=order,
        palette={label: PALETTE[label] for label in order},
        inner=None,
        cut=0,
        linewidth=0,
        alpha=0.18,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=delta,
        x="selector_label",
        y="accuracy_delta",
        order=order,
        color="#222222",
        size=2.0,
        alpha=0.22,
        jitter=0.24,
        ax=ax,
    )
    medians = delta.groupby("selector_label")["accuracy_delta"].median().reindex(order)
    for i, median in enumerate(medians):
        ax.plot([i - 0.22, i + 0.22], [median, median], color="#18181b", linewidth=2.4)
    ax.axhline(0, color="#111111", linewidth=1.2, linestyle="--")
    ax.set_xlabel("")
    ax.set_ylabel(r"Accuracy difference vs $\chi^2$")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", color="#d4d4d8", linewidth=0.8)
    save(fig, output_dir, "selector_delta_raincloud_vs_chi2")


def plot_selector_outcome_tiles(df, output_dir):
    ok = df[df["status"] == "ok"].copy()
    pivot = ok.pivot_table(index="dataset", columns="selector", values="accuracy", aggfunc="mean")
    if "chi2" not in pivot:
        return
    rows = []
    for selector in SELECTOR_ORDER:
        if selector == "chi2" or selector not in pivot:
            continue
        both = pivot[["chi2", selector]].dropna()
        diff = both[selector] - both["chi2"]
        rows.append(
            {
                "selector_label": SELECTOR_LABELS[selector],
                r"Alternative better": int((diff > 1e-12).sum()),
                "Tie": int((diff.abs() <= 1e-12).sum()),
                r"$\chi^2$ better": int((diff < -1e-12).sum()),
            }
        )
    outcome = pd.DataFrame(rows)
    if outcome.empty:
        return
    long = outcome.melt(id_vars="selector_label", var_name="Outcome", value_name="Datasets")
    long["share"] = long["Datasets"] / long.groupby("selector_label")["Datasets"].transform("sum")
    colors = {r"Alternative better": "#b85b5b", "Tie": "#a1a1aa", r"$\chi^2$ better": "#0f766e"}
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    y_positions = {label: i for i, label in enumerate(outcome["selector_label"])}
    x_offsets = {r"Alternative better": 0, "Tie": 1, r"$\chi^2$ better": 2}
    for _, row in long.iterrows():
        size = 2500 * max(row["share"], 0.03)
        ax.scatter(
            x_offsets[row["Outcome"]],
            y_positions[row["selector_label"]],
            s=size,
            marker="s",
            color=colors[row["Outcome"]],
            edgecolor="#fbfaf6",
            linewidth=2.0,
        )
        ax.text(
            x_offsets[row["Outcome"]],
            y_positions[row["selector_label"]],
            str(int(row["Datasets"])),
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            weight="bold",
        )
    ax.set_xticks(list(x_offsets.values()), list(x_offsets.keys()))
    ax.set_yticks(list(y_positions.values()), list(y_positions.keys()))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=10)
    ax.grid(False)
    save(fig, output_dir, "selector_outcome_tiles_vs_chi2")


def plot_selector_runtime_strips(summary, output_dir):
    if summary.empty:
        return
    working = summary.copy()
    working["selector_label"] = working["selector"].astype(str).map(SELECTOR_LABELS)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    y = np.arange(len(working))
    for i, row in working.reset_index(drop=True).iterrows():
        label = row["selector_label"]
        ax.plot(
            [row["median_feature_selection_time_s"], row["median_total_time_s"]],
            [i, i],
            color=PALETTE[label],
            linewidth=8,
            alpha=0.22,
            solid_capstyle="round",
        )
        ax.scatter(row["median_feature_selection_time_s"], i, s=70, color=PALETTE[label], edgecolor="#18181b", zorder=3)
        ax.scatter(row["median_total_time_s"], i, s=120, color=PALETTE[label], edgecolor="#18181b", zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(y, working["selector_label"])
    ax.set_xlabel("Seconds, log scale")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#d4d4d8", linewidth=0.8)
    save(fig, output_dir, "selector_runtime_strips")


def plot_selector_summary_matrix(summary, output_dir):
    if summary.empty:
        return
    working = summary.copy()
    working["selector_label"] = working["selector"].astype(str).map(SELECTOR_LABELS)
    working = working.set_index("selector_label")
    metrics = [
        ("Mean accuracy", "mean_accuracy", "{:.3f}"),
        ("Avg. rank", "average_rank", "{:.2f}"),
        ("Median runtime", "median_total_time_s", "{:.2f}s"),
        ("Median selection", "median_feature_selection_time_s", "{:.2f}s"),
    ]

    values = working[[col for _, col, _ in metrics]].astype(float)
    normalized = pd.DataFrame(index=values.index)
    for label, col, _ in metrics:
        series = values[col]
        lo = float(series.min())
        hi = float(series.max())
        if hi == lo:
            score = pd.Series(0.5, index=series.index)
        else:
            score = (series - lo) / (hi - lo)
        if col in {"average_rank", "median_total_time_s", "median_feature_selection_time_s"}:
            score = 1 - score
        normalized[label] = score

    annotations = pd.DataFrame(index=values.index)
    for label, col, fmt in metrics:
        annotations[label] = values[col].map(lambda value: fmt.format(value))

    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    sns.heatmap(
        normalized,
        cmap=SUMMARY_CMAP,
        vmin=0,
        vmax=1,
        linewidths=1.0,
        linecolor="#fbfaf6",
        annot=annotations,
        fmt="",
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=18)
    ax.tick_params(axis="y", rotation=0)
    save(fig, output_dir, "selector_overall_summary_matrix")


def plot_selector_efficiency_frontier(summary, output_dir):
    if summary.empty:
        return
    working = summary.copy()
    working["selector_label"] = working["selector"].astype(str).map(SELECTOR_LABELS)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    max_selection = max(float(working["median_feature_selection_time_s"].max()), 1e-9)
    for _, row in working.iterrows():
        label = row["selector_label"]
        size = 280 + 1050 * np.sqrt(row["median_feature_selection_time_s"] / max_selection)
        ax.scatter(
            row["median_total_time_s"],
            row["mean_accuracy"],
            s=size,
            color=PALETTE[label],
            edgecolor="#18181b",
            linewidth=0.9,
            alpha=0.88,
            zorder=3,
        )
        ax.annotate(
            label,
            (row["median_total_time_s"], row["mean_accuracy"]),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=9,
            weight="semibold",
        )
    ax.set_xscale("log")
    ax.set_xlim(working["median_total_time_s"].min() * 0.80, working["median_total_time_s"].max() * 1.75)
    ax.set_xlabel("Median end-to-end runtime, seconds (log scale)")
    ax.set_ylabel("Mean accuracy")
    ax.grid(axis="both", color="#d4d4d8", linewidth=0.8)
    save(fig, output_dir, "selector_efficiency_frontier")


def plot_selector_mean_accuracy_bars(summary, output_dir):
    if summary.empty:
        return
    working = summary.copy()
    working["selector_label"] = working["selector"].astype(str).map(SELECTOR_LABELS)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    y = np.arange(len(working))
    for i, row in working.reset_index(drop=True).iterrows():
        label = row["selector_label"]
        ax.hlines(i, 0, row["mean_accuracy"], color=PALETTE[label], linewidth=8, alpha=0.28)
        ax.scatter(row["mean_accuracy"], i, s=150, color=PALETTE[label], edgecolor="#18181b", zorder=3)
        ax.text(
            row["mean_accuracy"] + 0.006,
            i,
            f"{row['mean_accuracy']:.3f}",
            va="center",
            fontsize=9,
            weight="semibold",
        )
    ax.set_yticks(y, working["selector_label"])
    ax.set_xlim(max(0, working["mean_accuracy"].min() - 0.08), min(1.02, working["mean_accuracy"].max() + 0.08))
    ax.set_xlabel("Mean accuracy")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#d4d4d8", linewidth=0.8)
    save(fig, output_dir, "selector_mean_accuracy_lollipop")


def plot_results(results_csv, output_dir):
    raw = pd.read_csv(results_csv)
    df = complete_selector_cases(raw)
    if df.empty:
        raise SystemExit(
            "No complete selector-ablation datasets found. A dataset must have ok rows for "
            f"all selectors: {', '.join(SELECTOR_ORDER)}"
        )
    plots_dir = output_dir / "plots"
    if plots_dir.exists():
        for path in plots_dir.glob("*"):
            if path.is_file() and path.suffix.lower() in {".png", ".pdf"}:
                path.unlink()
    summary = summarize_results(df)
    summary.to_csv(output_dir / "selector_ablation_summary.csv", index=False)
    df.to_csv(output_dir / "selector_ablation_complete_cases.csv", index=False)
    plot_summary, _ = selector_summary_for_plots(df)
    style()
    plot_selector_accuracy_ecdf(df, plots_dir)
    plot_selector_radar(df, plot_summary, plots_dir)
    plot_selector_delta_raincloud(df, plots_dir)
    plot_selector_runtime_strips(summary, plots_dir)
    plot_selector_summary_matrix(summary, plots_dir)
    plot_selector_efficiency_frontier(summary, plots_dir)
    plot_selector_mean_accuracy_bars(summary, plots_dir)
    write_latex_table(summary, output_dir / "selector_ablation_table.tex")


def write_latex_table(summary, path):
    if summary.empty:
        return
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            f"{row['selector_label']} & {row['mean_accuracy']:.2f} & {row['average_rank']:.2f} & "
            f"{row['median_total_time_s']:.2f} & {row['median_feature_selection_time_s']:.2f} \\\\"
        )
    table = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\footnotesize",
            r"\caption{Ablation of the statistical feature-selection criterion in KFS-TUNE. All variants use the same random convolutional transform, feature budget, and RidgeClassifierCV; only the feature-scoring criterion changes. Accuracy and runtime are reported as aggregate performance summaries, and lower average rank is better.}",
            r"\label{tab:selector_ablation}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Selector & Mean accuracy & Avg. rank & Median runtime (s) & Median selection (s) \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text(table, encoding="utf-8")


def progress_counts(results_csv):
    if not results_csv.exists():
        return "No result rows written yet."
    df = pd.read_csv(results_csv)
    table = (
        df.pivot_table(index="selector", columns="status", values="dataset", aggfunc="nunique", fill_value=0)
        .reset_index()
        .sort_values("selector")
    )
    total = df.groupby("selector")["dataset"].nunique().rename("total_datasets").reset_index()
    return total.merge(table, on="selector", how="left").to_string(index=False)


def worker_command(args, dataset, selector, output_json):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-dataset",
        dataset,
        "--worker-selector",
        selector,
        "--worker-output-json",
        str(output_json),
        "--num-kernels",
        str(args.num_kernels),
        "--num-features",
        str(args.num_features),
        "--seed",
        str(args.seed),
    ]
    if args.ucr_path:
        cmd.extend(["--ucr-path", args.ucr_path])
    return cmd


def run_worker_subprocess(args, dataset, selector, output_dir):
    worker_dir = output_dir / "_worker"
    worker_dir.mkdir(parents=True, exist_ok=True)
    output_json = worker_dir / f"{dataset}__{selector}.json"
    if output_json.exists():
        output_json.unlink()
    try:
        completed = subprocess.run(
            worker_command(args, dataset, selector, output_json),
            cwd=str(ROOT),
            timeout=args.selector_timeout_s,
            text=True,
            capture_output=True,
        )
    except subprocess.TimeoutExpired:
        return SelectorResult(
            dataset=dataset,
            selector=selector,
            status="timeout",
            error=f"Timed out after {args.selector_timeout_s} seconds",
        )
    if output_json.exists():
        return SelectorResult(**json.loads(output_json.read_text(encoding="utf-8")))
    return SelectorResult(
        dataset=dataset,
        selector=selector,
        status="error",
        error=(
            f"Worker exited with code {completed.returncode} without writing output.\n"
            f"STDOUT:\n{completed.stdout[-4000:]}\nSTDERR:\n{completed.stderr[-4000:]}"
        ),
    )


def run_worker(args):
    x_train, y_train, x_test, y_test = load_ucr_dataset(args.worker_dataset, args.ucr_path)
    row = run_selector(args.worker_dataset, x_train, y_train, x_test, y_test, args)
    output_json = Path(args.worker_output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(asdict(row), indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run KFS-TUNE selector ablation: chi-square vs ANOVA-F vs mutual information vs random."
    )
    parser.add_argument("--datasets", nargs="+", default=UCR_BAKEOFF_DATASETS)
    parser.add_argument("--selectors", nargs="+", choices=SELECTOR_ORDER, default=SELECTOR_ORDER)
    parser.add_argument("--ucr-path", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "kfstune_selector_ablation"))
    parser.add_argument("--num-kernels", type=int, default=10000)
    parser.add_argument("--num-features", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selector-timeout-s", type=int, default=3600)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--rerun-statuses", nargs="+", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-dataset", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-selector", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-json", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.worker:
        run_worker(args)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / "selector_ablation_results.csv"
    summary_csv = output_dir / "selector_ablation_summary.csv"
    config_json = output_dir / "selector_ablation_config.json"

    if args.overwrite and results_csv.exists():
        results_csv.unlink()
    if args.plot_only:
        plot_results(results_csv, output_dir)
        print(f"Wrote plots to {output_dir / 'plots'}")
        return

    config_json.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    if args.skip_existing and args.rerun_statuses:
        remove_rerun_rows(results_csv, args.datasets, args.selectors, args.rerun_statuses)

    completed = existing_pairs(results_csv, args.rerun_statuses) if args.skip_existing else set()
    for dataset in args.datasets:
        print(f"\nLoading {dataset}")
        for selector in args.selectors:
            if (dataset, selector) in completed:
                print(f"  skipping {selector} (already has a row for {dataset})", flush=True)
                continue
            print(f"  running {selector}", flush=True)
            row = run_worker_subprocess(args, dataset, selector, output_dir)
            append_rows(results_csv, [row])
            if row.status == "ok":
                print(f"    accuracy={row.accuracy:.4f}, total={row.total_time_s:.2f}s", flush=True)
            else:
                print(f"    ERROR: {row.error.splitlines()[0] if row.error else 'unknown'}", flush=True)
            write_summary(results_csv, summary_csv)
            print(progress_counts(results_csv), flush=True)

    plot_results(results_csv, output_dir)
    print(f"\nWrote per-dataset results: {results_csv}")
    print(f"Wrote summary results: {summary_csv}")
    print(f"Wrote plots to: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
