#!/usr/bin/env python3
"""Standalone visualiser for MCP-RMS log files.

This script parses experiment log files produced by ``ExperimentRunner`` and
extracts structured data about job completion events, final makespans, and
potential error messages.  The resulting tables are exported as CSV files and
plotted using Matplotlib/Seaborn so analysts can quickly inspect progress and
identify instability such as segmentation faults.

The parser is intentionally self-contained – it does not import any package from
``src`` or ``experiments`` – which makes it suitable for ad-hoc investigations on
remote machines or CI workers that only have access to the raw logs.

Example::

    python scripts/plot_log_events.py \
        --log-file data/results/logs/benchmark_suite/instance.log \
        --output-dir data/results/log_analysis

Multiple ``--log-file`` arguments can be supplied and the script will merge
information across them.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LOGGER = logging.getLogger("log_visualiser")

# Regular expressions used for parsing the logs.
METHOD_PATTERN = re.compile(r"Method: (?P<method>[\w_]+)")
RUN_PATTERN = re.compile(r"Running (?P<method>[\w_]+) \(trial (?P<trial>\d+)\)")
JOB_PATTERN = re.compile(r"Job (?P<job>\d+) completed at time (?P<time>[0-9]+\.?[0-9]*)")
FINAL_PATTERN = re.compile(r"Final makespan: (?P<makespan>[0-9]+\.?[0-9]*)")
BEST_PATTERN = re.compile(r"Best cost = (?P<best>[0-9]+\.?[0-9]*)")
ERROR_PATTERN = re.compile(r"(ERROR|CRITICAL|Traceback|segmentation fault)", re.IGNORECASE)
WARNING_PATTERN = re.compile(r"WARNING")
TIMESTAMP_PATTERN = re.compile(r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


@dataclass
class ParsedLogs:
    """Container gathering structured information from logs."""

    events: pd.DataFrame
    summaries: pd.DataFrame
    issues: pd.DataFrame


def _normalise_log_paths(log_files: Sequence[Path]) -> List[Path]:
    paths: List[Path] = []
    for entry in log_files:
        if entry.is_file():
            paths.append(entry)
            continue
        if entry.is_dir():
            candidates = sorted(entry.rglob("*.log"))
            if not candidates:
                LOGGER.warning("Directory %s does not contain *.log files", entry)
            paths.extend(candidates)
            continue
        LOGGER.warning("Skipping %s because it does not exist", entry)
    return paths


def parse_logs(log_files: Sequence[Path]) -> ParsedLogs:
    """Parse log files and return structured data tables."""

    event_rows: List[dict] = []
    summary_rows: List[dict] = []
    issue_rows: List[dict] = []

    for log_path in _normalise_log_paths(log_files):
        LOGGER.info("Parsing %s", log_path)
        current_method: str | None = None
        current_trial: int | None = None
        line_index = 0

        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:  # pragma: no cover - defensive I/O
            LOGGER.error("Failed to read %s: %s", log_path, exc)
            continue

        for line in text.splitlines():
            line_index += 1
            method_match = METHOD_PATTERN.search(line)
            if method_match:
                current_method = method_match.group("method")
                LOGGER.debug("Detected method %s at line %d", current_method, line_index)

            run_match = RUN_PATTERN.search(line)
            if run_match:
                current_method = run_match.group("method")
                current_trial = int(run_match.group("trial"))
                LOGGER.debug(
                    "Detected run start: method=%s trial=%s line=%d",
                    current_method,
                    current_trial,
                    line_index,
                )

            job_match = JOB_PATTERN.search(line)
            if job_match and current_method is not None:
                event_rows.append(
                    {
                        "log_file": str(log_path),
                        "line": line_index,
                        "method": current_method,
                        "trial": current_trial,
                        "job_id": int(job_match.group("job")),
                        "completion_time": float(job_match.group("time")),
                    }
                )

            final_match = FINAL_PATTERN.search(line)
            if final_match and current_method is not None:
                summary_rows.append(
                    {
                        "log_file": str(log_path),
                        "line": line_index,
                        "method": current_method,
                        "trial": current_trial,
                        "metric": "final_makespan",
                        "value": float(final_match.group("makespan")),
                    }
                )

            best_match = BEST_PATTERN.search(line)
            if best_match and current_method is not None:
                summary_rows.append(
                    {
                        "log_file": str(log_path),
                        "line": line_index,
                        "method": current_method,
                        "trial": current_trial,
                        "metric": "best_cost",
                        "value": float(best_match.group("best")),
                    }
                )

            if ERROR_PATTERN.search(line) or WARNING_PATTERN.search(line):
                timestamp = None
                ts_match = TIMESTAMP_PATTERN.search(line)
                if ts_match:
                    timestamp = ts_match.group("timestamp")
                issue_rows.append(
                    {
                        "log_file": str(log_path),
                        "line": line_index,
                        "method": current_method,
                        "trial": current_trial,
                        "timestamp": timestamp,
                        "level": "ERROR" if ERROR_PATTERN.search(line) else "WARNING",
                        "message": line.strip(),
                    }
                )

    events_df = pd.DataFrame(event_rows)
    summaries_df = pd.DataFrame(summary_rows)
    issues_df = pd.DataFrame(issue_rows)

    if not events_df.empty:
        events_df.sort_values(["log_file", "method", "trial", "completion_time", "job_id"], inplace=True)
        events_df["cumulative_jobs"] = (
            events_df.groupby(["log_file", "method", "trial"], dropna=False).cumcount() + 1
        )

    if not summaries_df.empty:
        summaries_df.sort_values(["log_file", "method", "trial", "line"], inplace=True)

    if not issues_df.empty:
        issues_df.sort_values(["log_file", "line"], inplace=True)

    return ParsedLogs(events=events_df, summaries=summaries_df, issues=issues_df)


def _ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_tables(parsed: ParsedLogs, output_dir: Path) -> None:
    """Persist parsed dataframes as CSV files."""

    _ensure_output_dir(output_dir)

    if not parsed.events.empty:
        events_path = output_dir / "job_events.csv"
        parsed.events.to_csv(events_path, index=False)
        LOGGER.info("Wrote job events to %s", events_path)
    else:
        LOGGER.warning("No job completion events detected")

    if not parsed.summaries.empty:
        summaries_path = output_dir / "summary_metrics.csv"
        parsed.summaries.to_csv(summaries_path, index=False)
        LOGGER.info("Wrote summary metrics to %s", summaries_path)
    else:
        LOGGER.warning("No summary metrics detected")

    if not parsed.issues.empty:
        issues_path = output_dir / "log_issues.csv"
        parsed.issues.to_csv(issues_path, index=False)
        LOGGER.info("Wrote issue list to %s", issues_path)
    else:
        LOGGER.info("No warnings or errors found in logs")


def plot_job_progress(events: pd.DataFrame, output_dir: Path) -> None:
    """Plot cumulative job completions over time."""

    if events.empty:
        LOGGER.warning("Skipping job progress plot because no events are available")
        return

    _ensure_output_dir(output_dir)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 6))
    for (method, trial), group in events.groupby(["method", "trial"], dropna=False):
        label = method if trial is None else f"{method} (trial {trial})"
        ax.step(group["completion_time"], group["cumulative_jobs"], where="post", label=label)

    ax.set_xlabel("Completion time")
    ax.set_ylabel("Cumulative jobs finished")
    ax.set_title("Job completion progress")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()

    figure_path = output_dir / "job_progress.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    LOGGER.info("Saved job progress plot to %s", figure_path)


def plot_makespan_bars(summaries: pd.DataFrame, output_dir: Path) -> None:
    """Plot bar chart of final makespan by method/trial."""

    if summaries.empty:
        LOGGER.warning("Skipping makespan plot because no summary metrics are available")
        return

    makespans = summaries[summaries["metric"] == "final_makespan"].copy()
    if makespans.empty:
        LOGGER.warning("No final makespan records to plot")
        return

    _ensure_output_dir(output_dir)
    sns.set_theme(style="whitegrid")

    makespans["label"] = makespans.apply(
        lambda row: row["method"] if pd.isna(row["trial"]) else f"{row['method']} (trial {int(row['trial'])})",
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=makespans, x="value", y="label", ax=ax, palette="viridis")
    ax.set_xlabel("Final makespan")
    ax.set_ylabel("Method / trial")
    ax.set_title("Final makespan extracted from logs")
    fig.tight_layout()

    figure_path = output_dir / "makespan_bars.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    LOGGER.info("Saved makespan bar plot to %s", figure_path)


def plot_best_cost_lines(summaries: pd.DataFrame, output_dir: Path) -> None:
    """Plot the evolution of reported best costs when available."""

    best_costs = summaries[summaries["metric"] == "best_cost"].copy()
    if best_costs.empty:
        LOGGER.info("Skipping best-cost plot because logs did not report this metric")
        return

    _ensure_output_dir(output_dir)
    sns.set_theme(style="whitegrid")

    best_costs["line_order"] = best_costs.groupby(["method", "trial"], dropna=False).cumcount()

    fig, ax = plt.subplots(figsize=(10, 6))
    for (method, trial), group in best_costs.groupby(["method", "trial"], dropna=False):
        label = method if trial is None else f"{method} (trial {trial})"
        ax.plot(group["line_order"], group["value"], marker="o", label=label)

    ax.set_xlabel("Iteration index in log")
    ax.set_ylabel("Best cost")
    ax.set_title("Reported best-cost progression")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()

    figure_path = output_dir / "best_cost_progress.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    LOGGER.info("Saved best-cost progression plot to %s", figure_path)


def generate_plots(parsed: ParsedLogs, output_dir: Path) -> None:
    """Create all visualisations from parsed logs."""

    figures_dir = _ensure_output_dir(output_dir / "figures")
    plot_job_progress(parsed.events, figures_dir)
    plot_makespan_bars(parsed.summaries, figures_dir)
    plot_best_cost_lines(parsed.summaries, figures_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise MCP-RMS experiment logs")
    parser.add_argument(
        "--log-file",
        "-l",
        type=Path,
        action="append",
        required=True,
        help="Path to a log file or directory containing logs (can be repeated)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/log_analysis"),
        help="Directory where CSV files and plots will be stored",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity for the parser itself",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    parsed = parse_logs(args.log_file)
    export_tables(parsed, args.output_dir)
    generate_plots(parsed, args.output_dir)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
