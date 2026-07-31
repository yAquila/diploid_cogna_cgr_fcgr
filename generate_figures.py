"""Generate IEEE report figures from classification summary CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"

CSV_FILES = {
    "cgr_empr_balanced": ROOT / "cgr_empr_balanced_summary.csv",
    "cgr_empr_imbalanced": ROOT / "cgr_empr_imbalanced_summary.csv",
    "fcgr_balanced": ROOT / "fcgr_dct_svd_balanced_summary.csv",
    "fcgr_imbalanced": ROOT / "fcgr_dct_svd_imbalanced_summary.csv",
}

METHOD_STYLE = {
    "CGR + EMPR": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "DCT": {"color": "#2ca02c", "marker": "s", "linestyle": "-"},
    "DCT + SVD": {"color": "#ff7f0e", "marker": "^", "linestyle": "--"},
    "SVD": {"color": "#d62728", "marker": "x", "linestyle": ":"},
}

METRICS = [
    ("OA_percent_mean", "OA_percent_std", "Overall Accuracy (%)"),
    ("F1_mean", "F1_std", "F1 Score"),
    ("Balanced_accuracy_mean", "Balanced_accuracy_std", "Balanced Accuracy"),
    ("AUC_mean", "AUC_std", "AUC"),
]


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Method"] = df["Method"].astype(str)
    return df


def combine_protocol(cgr_df: pd.DataFrame, fcgr_df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([cgr_df, fcgr_df], ignore_index=True)


def plot_metric_curves(df: pd.DataFrame, title_prefix: str, output_stem: str) -> None:
    methods = list(METHOD_STYLE.keys())
    methods = [m for m in methods if m in set(df["Method"])]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True)
    axes = axes.ravel()

    for ax, (mean_col, std_col, ylabel) in zip(axes, METRICS):
        for method in methods:
            subset = df[df["Method"] == method].sort_values("S")
            style = METHOD_STYLE[method]
            y = subset[mean_col].to_numpy(dtype=float).copy()
            yerr = subset[std_col].to_numpy(dtype=float).copy()
            if mean_col == "Balanced_accuracy_mean":
                y *= 100.0
                yerr *= 100.0
                ylabel = "Balanced Accuracy (%)"
            ax.errorbar(
                subset["S"],
                y,
                yerr=yerr,
                label=method,
                capsize=3,
                linewidth=1.8,
                markersize=6,
                **style,
            )
        ax.set_xlabel("Training Subjects per Class (S)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(df["S"].unique()))
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_ylim(bottom=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{title_prefix}: Learning Curves (10 Runs, Mean ± Std)", y=1.06, fontsize=13)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{output_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_summary_table(df: pd.DataFrame, s_values: list[int]) -> pd.DataFrame:
    rows = []
    for s in s_values:
        for method in sorted(df["Method"].unique()):
            row = df[(df["Method"] == method) & (df["S"] == s)].iloc[0]
            entry = {
                "S": s,
                "Method": method,
                "OA (%)": f"{row['OA_percent_mean']:.1f} ± {row['OA_percent_std']:.1f}",
                "Bal. Acc.": f"{row['Balanced_accuracy_mean'] * 100:.1f} ± {row['Balanced_accuracy_std'] * 100:.1f}",
                "F1": f"{row['F1_mean']:.3f} ± {row['F1_std']:.3f}",
                "AUC": f"{row['AUC_mean']:.3f} ± {row['AUC_std']:.3f}",
            }
            if "MCC_mean" in row.index and not pd.isna(row["MCC_mean"]):
                entry["MCC"] = f"{row['MCC_mean']:.3f} ± {row['MCC_std']:.3f}"
            rows.append(entry)
    return pd.DataFrame(rows)


def plot_small_large_bars(balanced: pd.DataFrame, imbalanced: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    configs = [
        (balanced, "Balanced (S=10 and S=50)", "balanced_bar"),
        (imbalanced, "Imbalanced 1:4 (S=10 and S=50)", "imbalanced_bar"),
    ]

    for ax, (df, title, stem) in zip(axes, configs):
        plot_df = df[df["S"].isin([10, 50])].copy()
        methods = [m for m in METHOD_STYLE if m in set(plot_df["Method"])]
        x = np.arange(len(methods))
        width = 0.35

        for idx, s in enumerate([10, 50]):
            values = []
            errors = []
            for method in methods:
                row = plot_df[(plot_df["Method"] == method) & (plot_df["S"] == s)].iloc[0]
                values.append(row["F1_mean"])
                errors.append(row["F1_std"])
            offset = -width / 2 if idx == 0 else width / 2
            ax.bar(
                x + offset,
                values,
                width,
                yerr=errors,
                capsize=3,
                label=f"S={s}",
                alpha=0.9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.set_ylabel("F1 Score")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.legend(frameon=False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"f1_comparison_bar.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cgr_bal = load_csv(CSV_FILES["cgr_empr_balanced"])
    cgr_imb = load_csv(CSV_FILES["cgr_empr_imbalanced"])
    fcgr_bal = load_csv(CSV_FILES["fcgr_balanced"])
    fcgr_imb = load_csv(CSV_FILES["fcgr_imbalanced"])

    balanced = combine_protocol(cgr_bal, fcgr_bal)
    imbalanced = combine_protocol(cgr_imb, fcgr_imb)

    plot_metric_curves(balanced, "Balanced Protocol", "balanced_learning_curves")
    plot_metric_curves(imbalanced, "Imbalanced 1:4 Protocol", "imbalanced_learning_curves")
    plot_small_large_bars(balanced, imbalanced)

    build_summary_table(balanced, [10, 50]).to_csv(FIG_DIR / "balanced_table_s10_s50.csv", index=False)
    build_summary_table(imbalanced, [10, 50]).to_csv(FIG_DIR / "imbalanced_table_s10_s50.csv", index=False)

    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
