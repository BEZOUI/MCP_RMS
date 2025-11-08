#!/usr/bin/env python3
"""Standalone script to analyse MCP-RMS experiment outputs and plot figures.

The script expects the experiment runner to have produced JSON summaries in
``data/results`` (or a custom directory passed through ``--input-dir``).
Each JSON file should contain the list of per-instance result dictionaries
returned by :class:`ExperimentRunner`.  The script consolidates the aggregated
statistics and the per-trial raw metrics, computes textual summaries, and
exports a collection of publication-ready plots to ``<output>/figures``.  It
also scans per-trial log files to generate a ``log_summary.csv`` report that
highlights warnings, errors, and the last line observed in every run – useful
when hunting sporadic crashes.

Usage
-----
    python scripts/plot_results.py \
        --input-dir data/results \
        --output-dir data/results/standalone_figures

The command is safe to run repeatedly: missing files are skipped gracefully and
plots are overwritten when the underlying data change.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


LOGGER = logging.getLogger("standalone_results_plotter")
DEFAULT_METRICS: Tuple[str, ...] = (
    "makespan_mean",
    "tardiness_mean",
    "energy_mean",
    "time_mean",
)
TRIAL_METRIC_MAP = {
    "makespan_mean": "makespan",
    "tardiness_mean": "total_tardiness",
    "energy_mean": "energy_consumption",
    "time_mean": "time",
}


@dataclass
class LoadedResults:
    """Container for the consolidated experiment results."""

    aggregated: pd.DataFrame
    trials: pd.DataFrame


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def _ensure_dataframe(df: pd.DataFrame, numeric_columns: Iterable[str]) -> pd.DataFrame:
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _load_results_file(path: Path) -> List[Dict]:
    """Load a single JSON file and normalise the structure."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        LOGGER.warning("Skipping %s: invalid JSON (%s)", path, exc)
        return []

    if isinstance(payload, dict):
        # Legacy format that wraps results in a top-level key.
        for key in ("results", "instances", "data"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
        return []

    if isinstance(payload, list):
        return payload

    LOGGER.warning("Skipping %s: unsupported payload type %s", path, type(payload))
    return []


def _relative_parent(input_dir: Path, file_path: Path) -> str:
    try:
        relative = file_path.relative_to(input_dir)
    except ValueError:
        return file_path.parent.name
    parts = relative.parts
    if len(parts) <= 1:
        return "root"
    return str(Path(*parts[:-1]))


def _iter_result_files(input_dir: Path, recursive: bool) -> Sequence[Path]:
    pattern = "**/*_results.json" if recursive else "*_results.json"
    files = sorted(input_dir.glob(pattern))
    if not files:
        LOGGER.warning("No JSON result files found in %s (recursive=%s)", input_dir, recursive)
    return files


def load_results(input_dir: Path, recursive: bool = True) -> LoadedResults:
    """Load and flatten aggregated and per-trial results from JSON files."""

    aggregated_rows: List[Dict] = []
    trial_rows: List[Dict] = []

    for json_file in _iter_result_files(input_dir, recursive):
        suite = _relative_parent(input_dir, json_file)
        LOGGER.info("Reading results from %s", json_file)
        instances = _load_results_file(json_file)

        for instance in instances:
            instance_name = instance.get("instance_name") or instance.get("instance")
            num_machines = instance.get("num_machines")
            num_jobs = instance.get("num_jobs")
            instance_size = (
                f"{num_machines}x{num_jobs}" if num_machines and num_jobs else None
            )

            methods = instance.get("methods", {})
            for method_name, method_data in methods.items():
                base_record = {
                    "suite": suite,
                    "suite_file": json_file.name,
                    "instance": instance_name,
                    "method": method_name,
                    "num_machines": num_machines,
                    "num_jobs": num_jobs,
                    "instance_size": instance_size,
                    "success_rate": method_data.get("success_rate"),
                }

                aggregated_rows.append(
                    {
                        **base_record,
                        "makespan_mean": method_data.get("makespan_mean"),
                        "makespan_std": method_data.get("makespan_std"),
                        "makespan_min": method_data.get("makespan_min"),
                        "makespan_max": method_data.get("makespan_max"),
                        "tardiness_mean": method_data.get("tardiness_mean"),
                        "energy_mean": method_data.get("energy_mean"),
                        "time_mean": method_data.get("time_mean"),
                    }
                )

                for trial_idx, trial in enumerate(method_data.get("raw_results", [])):
                    metrics = trial.get("metrics") or {}
                    trial_rows.append(
                        {
                            **base_record,
                            "trial": trial_idx,
                            "success": trial.get("success", bool(metrics)),
                            "time": trial.get("time"),
                            "error": trial.get("error"),
                            "makespan": metrics.get("makespan"),
                            "total_tardiness": metrics.get("total_tardiness"),
                            "energy_consumption": metrics.get("energy_consumption"),
                        }
                    )

    aggregated_df = _ensure_dataframe(
        pd.DataFrame(aggregated_rows),
        DEFAULT_METRICS
        + ("makespan_std", "makespan_min", "makespan_max", "success_rate"),
    )
    trials_df = _ensure_dataframe(
        pd.DataFrame(trial_rows),
        ("time", "makespan", "total_tardiness", "energy_consumption"),
    )

    return LoadedResults(aggregated=aggregated_df, trials=trials_df)


# ---------------------------------------------------------------------------
# Analytical helpers
# ---------------------------------------------------------------------------

def describe_best_methods(aggregated: pd.DataFrame) -> pd.DataFrame:
    """Return the best-performing method per instance for each metric."""

    if aggregated.empty:
        return pd.DataFrame()

    summaries: List[pd.DataFrame] = []
    metric_preferences = {
        "makespan_mean": "min",
        "tardiness_mean": "min",
        "energy_mean": "min",
        "time_mean": "min",
        "success_rate": "max",
    }

    for metric, mode in metric_preferences.items():
        if metric not in aggregated.columns:
            continue
        subset = aggregated.dropna(subset=[metric])
        if subset.empty:
            continue

        if mode == "min":
            idx = subset.groupby("instance")[metric].idxmin()
        else:
            idx = subset.groupby("instance")[metric].idxmax()

        best_rows = subset.loc[idx].copy()
        best_rows["metric"] = metric
        best_rows.rename(columns={metric: "value"}, inplace=True)
        summaries.append(best_rows[["instance", "method", "metric", "value"]])

    if not summaries:
        return pd.DataFrame()

    leaderboard = pd.concat(summaries, ignore_index=True)
    leaderboard.sort_values(["metric", "value"], inplace=True)
    return leaderboard


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _initialise_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.figsize": (12, 7),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def _save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{filename}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved figure %s", path)
    return path


def plot_method_bars(aggregated: pd.DataFrame, metric: str, output_dir: Path) -> Path | None:
    if aggregated.empty or metric not in aggregated.columns:
        return None

    subset = aggregated.dropna(subset=[metric])
    if subset.empty:
        return None

    grouped = subset.groupby("method")[metric].agg(["mean", "std", "count"])
    grouped.sort_values("mean", ascending=True, inplace=True)

    fig, ax = plt.subplots()
    ax.bar(
        x=np.arange(len(grouped)),
        height=grouped["mean"],
        yerr=grouped["std"],
        capsize=5,
        color="steelblue",
        alpha=0.85,
    )
    ax.set_xticks(np.arange(len(grouped)))
    ax.set_xticklabels(grouped.index, rotation=45, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Average {metric.replace('_', ' ').title()} by Method")
    ax.grid(axis="y", alpha=0.3)

    return _save_figure(fig, output_dir, f"bar_{metric}")


def plot_trial_distributions(trials: pd.DataFrame, metric: str, output_dir: Path) -> Path | None:
    if trials.empty or metric not in trials.columns:
        return None

    subset = trials.dropna(subset=[metric])
    if subset.empty:
        return None

    fig, ax = plt.subplots()
    sns.boxplot(data=subset, x="method", y=metric, ax=ax)
    ax.set_xlabel("Method")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Distribution of {metric.replace('_', ' ').title()} by Method")
    ax.tick_params(axis="x", rotation=45)

    return _save_figure(fig, output_dir, f"box_{metric}")


def plot_success_heatmap(aggregated: pd.DataFrame, output_dir: Path) -> Path | None:
    if aggregated.empty or "success_rate" not in aggregated.columns:
        return None

    pivot = (
        aggregated.dropna(subset=["success_rate"])
        .pivot_table(
            index="instance",
            columns="method",
            values="success_rate",
            aggfunc="mean",
        )
    )
    if pivot.empty:
        return None

    fig, ax = plt.subplots(figsize=(max(12, pivot.shape[1] * 1.2), max(6, pivot.shape[0] * 0.5)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar_kws={"label": "Success Rate"})
    ax.set_xlabel("Method")
    ax.set_ylabel("Instance")
    ax.set_title("Success Rate by Instance and Method")

    return _save_figure(fig, output_dir, "heatmap_success_rate")


def plot_runtime_vs_quality(aggregated: pd.DataFrame, output_dir: Path) -> Path | None:
    if aggregated.empty:
        return None

    subset = aggregated.dropna(subset=["makespan_mean", "time_mean"])
    if subset.empty:
        return None

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=subset,
        x="time_mean",
        y="makespan_mean",
        hue="method",
        style="method",
        ax=ax,
        s=80,
    )
    ax.set_xlabel("Average Runtime (s)")
    ax.set_ylabel("Average Makespan")
    ax.set_title("Runtime vs. Makespan")
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    return _save_figure(fig, output_dir, "scatter_runtime_vs_makespan")


def plot_makespan_vs_size(aggregated: pd.DataFrame, output_dir: Path) -> Path | None:
    required_cols = {"num_jobs", "num_machines", "makespan_mean"}
    if aggregated.empty or not required_cols.issubset(aggregated.columns):
        return None

    subset = aggregated.dropna(subset=list(required_cols))
    if subset.empty:
        return None

    subset = subset.copy()
    subset["problem_size"] = subset["num_jobs"] * subset["num_machines"]

    fig, ax = plt.subplots()
    sns.lineplot(data=subset, x="problem_size", y="makespan_mean", hue="method", marker="o", ax=ax)
    ax.set_xlabel("Problem Size (machines × jobs)")
    ax.set_ylabel("Average Makespan")
    ax.set_title("Makespan Growth with Problem Size")
    ax.grid(alpha=0.3)

    return _save_figure(fig, output_dir, "line_makespan_vs_problem_size")


def plot_metric_grid(aggregated: pd.DataFrame, metric: str, output_dir: Path) -> Path | None:
    if aggregated.empty or metric not in aggregated.columns or "suite" not in aggregated.columns:
        return None

    subset = aggregated.dropna(subset=[metric])
    if subset.empty:
        return None

    grid = sns.catplot(
        data=subset,
        x="method",
        y=metric,
        hue="method",
        col="suite",
        kind="bar",
        col_wrap=3,
        sharey=False,
        height=4,
        aspect=1.2,
    )
    grid.set_titles("{col_name}")
    grid.set_xlabels("Method")
    grid.set_ylabels(metric.replace("_", " ").title())
    for ax in grid.axes.flatten():
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"grid_{metric}.png"
    grid.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(grid.fig)
    LOGGER.info("Saved grid figure %s", path)
    return path


PLOTTERS = (
    lambda agg, trials, out: plot_success_heatmap(agg, out),
    lambda agg, trials, out: plot_runtime_vs_quality(agg, out),
    lambda agg, trials, out: plot_makespan_vs_size(agg, out),
)


LOG_ERROR_PATTERN = re.compile(r"(ERROR|Exception|Traceback|segmentation fault)", re.IGNORECASE)
LOG_WARNING_PATTERN = re.compile(r"WARNING", re.IGNORECASE)
TIMESTAMP_PATTERN = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def _iter_log_files(input_dir: Path, recursive: bool) -> Sequence[Path]:
    pattern = "**/*.log" if recursive else "*.log"
    return sorted(input_dir.glob(pattern))


def summarise_logs(input_dir: Path, output_dir: Path, recursive: bool = True) -> pd.DataFrame:
    rows: List[Dict] = []

    for log_path in _iter_log_files(input_dir, recursive):
        suite = _relative_parent(input_dir, log_path)
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:  # pragma: no cover - defensive
            LOGGER.warning("Unable to read log %s (%s)", log_path, exc)
            continue

        lines = [line for line in text.splitlines() if line.strip()]
        error_matches = sum(1 for line in lines if LOG_ERROR_PATTERN.search(line))
        warning_matches = sum(1 for line in lines if LOG_WARNING_PATTERN.search(line))
        timestamps = [TIMESTAMP_PATTERN.search(line).group("ts") for line in lines if TIMESTAMP_PATTERN.search(line)]

        rows.append(
            {
                "suite": suite,
                "log_file": str(log_path.relative_to(input_dir)),
                "num_lines": len(lines),
                "warnings": warning_matches,
                "errors": error_matches,
                "first_timestamp": timestamps[0] if timestamps else None,
                "last_timestamp": timestamps[-1] if timestamps else None,
                "last_line": lines[-1] if lines else "",
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        LOGGER.warning("No log files found for summarisation")
        return df

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "log_summary.csv"
    df.sort_values(["suite", "log_file"], inplace=True)
    df.to_csv(report_path, index=False)
    LOGGER.info("Wrote log summary to %s", report_path)
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone MCP-RMS results visualiser")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/results"),
        help="Directory containing *_results.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/standalone_figures"),
        help="Directory where plots will be stored",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive search for JSON and log files",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export aggregated.csv and trials.csv alongside plots",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    _initialise_style()

    recursive = not args.no_recursive
    results = load_results(args.input_dir, recursive=recursive)
    aggregated, trials = results.aggregated, results.trials

    if aggregated.empty and trials.empty:
        LOGGER.warning("No results loaded – nothing to plot")
        return

    figures_dir = args.output_dir / "figures"

    for metric in DEFAULT_METRICS:
        plot_method_bars(aggregated, metric, figures_dir)
        trial_metric = TRIAL_METRIC_MAP.get(metric)
        if trial_metric:
            plot_trial_distributions(trials, trial_metric, figures_dir)
        plot_metric_grid(aggregated, metric, figures_dir)

    for plotter in PLOTTERS:
        plotter(aggregated, trials, figures_dir)

    leaderboard = describe_best_methods(aggregated)
    if not leaderboard.empty:
        summary_path = args.output_dir / "best_methods.csv"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        leaderboard.to_csv(summary_path, index=False)
        LOGGER.info("Wrote best-method summary to %s", summary_path)
    else:
        LOGGER.warning("Unable to compute best-method summary")

    if args.export_csv:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        aggregated.to_csv(args.output_dir / "aggregated.csv", index=False)
        trials.to_csv(args.output_dir / "trials.csv", index=False)
        LOGGER.info("Exported consolidated CSV tables to %s", args.output_dir)

    summarise_logs(
        args.input_dir,
        args.output_dir / "log_reports",
        recursive=recursive,
    )


if __name__ == "__main__":
    main()
