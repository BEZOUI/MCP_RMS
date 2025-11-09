#!/usr/bin/env python3
"""Monolithic runner for the Adaptive MCP-RMS method with cinematic visuals.

This standalone script focuses exclusively on the Adaptive MCP workflow. It
builds a compact RMS instance, executes the solver with the local Ollama LLM
(or a deterministic heuristic fallback), and exports a suite of artefacts:
    * detailed schedule and metrics tables
    * a publication-ready Gantt chart
    * machine health heatmaps
    * job completion trajectories
    * interactive Plotly dashboards

Usage example (heuristic only):
    python monolith_adaptive_mcp.py --disable-llm --output-dir runs/monolith

Usage example (with local Ollama auto-selection):
    python monolith_adaptive_mcp.py --output-dir runs/ollama
"""
from __future__ import annotations

import argparse
import json
import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

MATPLOTLIB_IMPORT_ERROR: Optional[Exception]
SEABORN_IMPORT_ERROR: Optional[Exception]

MATPLOTLIB_IMPORT_ERROR = None
SEABORN_IMPORT_ERROR = None
HAS_MATPLOTLIB = False
HAS_SEABORN = False

try:
    import matplotlib

    matplotlib.use("Agg")  # Headless backend for reliable rendering
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception as exc:  # pragma: no cover - depends on local environment
    MATPLOTLIB_IMPORT_ERROR = exc
    plt = None  # type: ignore
    matplotlib = None  # type: ignore
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception as exc:  # pragma: no cover - depends on local environment
    SEABORN_IMPORT_ERROR = exc
    sns = None  # type: ignore
    HAS_SEABORN = False

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from src.environment import RMSEnvironment
from src.memory import MemorySystem
from src.mcp_server import MCPServer

try:  # Prefer local Ollama client when available
    from src.llm_client import LLMClient
except Exception:  # pragma: no cover - defensive in minimal installs
    LLMClient = None  # type: ignore


class DisabledLLMClient:
    """Simple stand-in that mimics the LLM interface when disabled."""

    def __init__(self, reason: str = "Local LLM disabled for this run") -> None:
        self.reason = reason
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "avg_response_time": 0.0,
            "disabled": True,
            "reason": reason,
        }

    def generate_response(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            "success": False,
            "error": self.reason,
            "response": None,
            "fatal": False,
        }

    def parse_action_from_response(self, _response: str) -> List[Dict[str, Any]]:
        return []

    def list_available_models(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:  # pragma: no cover
        return []

    def get_statistics(self) -> Dict[str, Any]:
        return dict(self.stats)


# ---------------------------------------------------------------------------
# Logging & configuration helpers
# ---------------------------------------------------------------------------


def configure_logging(log_path: Path, verbose: bool = False) -> None:
    """Configure both console and file logging."""

    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    console_level = logging.INFO if not verbose else logging.DEBUG
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(console_level)
    root_logger.addHandler(console_handler)

    logging.getLogger(__name__).info("Logging initialized -> %s", log_path)

    if MATPLOTLIB_IMPORT_ERROR is not None:
        logging.getLogger(__name__).warning(
            "Matplotlib unavailable – static PNG charts will be replaced with HTML alternatives (%s)",
            MATPLOTLIB_IMPORT_ERROR,
        )
    elif not HAS_MATPLOTLIB:
        logging.getLogger(__name__).warning(
            "Matplotlib support unknown – static PNG charts may be skipped"
        )

    if SEABORN_IMPORT_ERROR is not None:
        logging.getLogger(__name__).warning(
            "Seaborn unavailable – heatmaps and line plots will fall back to Plotly HTML (%s)",
            SEABORN_IMPORT_ERROR,
        )


# ---------------------------------------------------------------------------
# Data extraction utilities
# ---------------------------------------------------------------------------


def build_schedule_dataframe(schedule_events: List[Dict[str, Any]]) -> pd.DataFrame:
    """Normalise raw environment schedule events into a tidy DataFrame."""

    if not schedule_events:
        return pd.DataFrame(
            columns=[
                "event",
                "time",
                "job_id",
                "operation_id",
                "machine_id",
                "completion_time",
                "processing_time",
                "from_config",
                "to_config",
            ]
        )

    df = pd.DataFrame(schedule_events)
    df = df.sort_values("time").reset_index(drop=True)

    # Enrich for plotting convenience
    if "completion_time" in df.columns:
        df["duration"] = df["completion_time"] - df["time"]
    else:
        df["duration"] = np.nan

    df["machine_label"] = df["machine_id"].apply(
        lambda x: f"Machine {int(x)}" if not pd.isna(x) else "N/A"
    )
    df["job_label"] = df["job_id"].apply(
        lambda x: f"Job {int(x)}" if not pd.isna(x) else "—"
    )
    df["operation_label"] = df.apply(
        lambda row: f"Op {int(row['operation_id'])}" if not pd.isna(row.get("operation_id")) else "",
        axis=1,
    )

    return df


def extract_machine_stats(env: RMSEnvironment) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    horizon = max(env.current_time, 1.0)

    for machine in env.machines:
        utilization = machine.total_processing_time / horizon
        rows.append(
            {
                "machine_id": machine.machine_id,
                "machine": f"Machine {machine.machine_id}",
                "utilization": utilization,
                "energy": machine.total_energy,
                "reconfigurations": machine.reconfiguration_count,
                "next_available": machine.next_available_time,
            }
        )

    df = pd.DataFrame(rows)
    return df


def extract_job_stats(env: RMSEnvironment) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for job in env.completed_jobs:
        rows.append(
            {
                "job_id": job.job_id,
                "arrival": job.arrival_time,
                "due_date": job.due_date,
                "completion": getattr(job, "completion_time", np.nan),
                "flowtime": getattr(job, "flowtime", np.nan),
                "tardiness": getattr(job, "tardiness", 0.0),
                "priority": job.priority,
                "operations": len(job.operations),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def _ensure_html_path(output_path: Path, suffix: str = ".html") -> Path:
    return output_path if output_path.suffix == suffix else output_path.with_suffix(suffix)


def _plot_machine_heatmap_plotly(machine_df: pd.DataFrame, output_path: Path) -> Optional[Path]:
    if machine_df.empty:
        return None

    html_path = _ensure_html_path(output_path)
    heatmap_df = machine_df.set_index("machine")[
        ["utilization", "energy", "reconfigurations"]
    ]
    z = heatmap_df.to_numpy()
    fig = go.Figure(
        data=
        go.Heatmap(
            z=z,
            x=heatmap_df.columns,
            y=heatmap_df.index,
            colorscale="Viridis",
            hovertemplate="Machine=%{y}<br>%{x}=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Machine Vital Signs (interactive)",
        xaxis_title="Metric",
        yaxis_title="Machine",
    )
    pio.write_html(fig, html_path, include_plotlyjs="cdn", auto_open=False)
    logging.getLogger(__name__).info(
        "Saved interactive machine heatmap -> %s", html_path
    )
    return html_path


def _plot_job_completion_plotly(job_df: pd.DataFrame, output_path: Path) -> Optional[Path]:
    if job_df.empty or job_df["completion"].isna().all():
        return None

    html_path = _ensure_html_path(output_path)
    timeline = job_df.sort_values("completion").reset_index(drop=True)
    timeline["completed_jobs"] = np.arange(1, len(timeline) + 1)
    fig = go.Figure(
        data=
        go.Scatter(
            x=timeline["completion"],
            y=timeline["completed_jobs"],
            mode="lines+markers",
            line=dict(color="#2E86AB", width=3),
            fill="tozeroy",
        )
    )
    fig.update_layout(
        title="Throughput Trajectory – Adaptive MCP-RMS (interactive)",
        xaxis_title="Time",
        yaxis_title="Cumulative completed jobs",
    )
    pio.write_html(fig, html_path, include_plotlyjs="cdn", auto_open=False)
    logging.getLogger(__name__).info(
        "Saved interactive throughput trajectory -> %s", html_path
    )
    return html_path


def plot_gantt(schedule_df: pd.DataFrame, output_path: Path) -> Optional[Path]:
    if schedule_df.empty:
        logging.getLogger(__name__).warning("No schedule events recorded – skipping Gantt chart")
        return None

    ops_df = schedule_df[schedule_df["event"] == "operation_start"].copy()
    if ops_df.empty:
        logging.getLogger(__name__).warning("No operation events present – skipping Gantt chart")
        return None

    if not HAS_MATPLOTLIB or not HAS_SEABORN or plt is None or sns is None:
        html_path = _ensure_html_path(output_path)
        logging.getLogger(__name__).warning(
            "Matplotlib/Seaborn unavailable – exporting interactive Gantt instead -> %s",
            html_path,
        )
        return plot_interactive_timeline(schedule_df, html_path)

    ops_df["duration"] = ops_df["duration"].fillna(
        ops_df["completion_time"] - ops_df["time"]
    )

    palette = sns.color_palette("Spectral", n_colors=ops_df["job_id"].nunique())
    color_map = {
        job_id: palette[idx % len(palette)]
        for idx, job_id in enumerate(sorted(ops_df["job_id"].unique()))
    }

    fig, ax = plt.subplots(figsize=(14, 7))

    for _, row in ops_df.iterrows():
        job_color = color_map.get(row["job_id"], (0.2, 0.2, 0.2))
        ax.barh(
            row["machine_label"],
            row["duration"],
            left=row["time"],
            color=job_color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.text(
            row["time"] + row["duration"] / 2,
            row["machine_label"],
            f"J{int(row['job_id'])}-O{int(row['operation_id'])}",
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_title("Adaptive MCP-RMS Schedule (Gantt)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_map[jid])
        for jid in color_map
    ]
    labels = [f"Job {jid}" for jid in color_map]
    ax.legend(handles, labels, bbox_to_anchor=(1.04, 1), loc="upper left", title="Jobs")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logging.getLogger(__name__).info("Saved cinematic Gantt chart -> %s", output_path)
    return output_path


def plot_machine_heatmap(machine_df: pd.DataFrame, output_path: Path) -> Optional[Path]:
    if machine_df.empty:
        return None

    if not HAS_MATPLOTLIB or not HAS_SEABORN or plt is None or sns is None:
        logging.getLogger(__name__).warning(
            "Matplotlib/Seaborn unavailable – exporting interactive heatmap instead"
        )
        return _plot_machine_heatmap_plotly(machine_df, output_path)

    heatmap_df = machine_df.set_index("machine")[
        ["utilization", "energy", "reconfigurations"]
    ]

    scaled = heatmap_df.copy()
    scaled["utilization"] = scaled["utilization"].clip(0, 1)
    if scaled["energy"].max() > 0:
        scaled["energy"] = scaled["energy"] / scaled["energy"].max()
    if scaled["reconfigurations"].max() > 0:
        scaled["reconfigurations"] = (
            scaled["reconfigurations"] / scaled["reconfigurations"].max()
        )

    plt.figure(figsize=(10, max(4, len(heatmap_df) * 0.4)))
    sns.heatmap(
        scaled,
        annot=heatmap_df.round({"utilization": 2, "energy": 1, "reconfigurations": 0}),
        fmt="",
        cmap="mako",
        cbar_kws={"label": "Normalised intensity"},
    )
    plt.title("Machine Vital Signs – Utilisation, Energy & Reconfigurations")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    logging.getLogger(__name__).info("Saved machine heatmap -> %s", output_path)
    return output_path


def plot_job_completion(job_df: pd.DataFrame, output_path: Path) -> Optional[Path]:
    if job_df.empty or job_df["completion"].isna().all():
        logging.getLogger(__name__).warning("No completed jobs available – skipping trajectory plot")
        return None

    if not HAS_MATPLOTLIB or not HAS_SEABORN or plt is None or sns is None:
        logging.getLogger(__name__).warning(
            "Matplotlib/Seaborn unavailable – exporting interactive throughput plot instead"
        )
        return _plot_job_completion_plotly(job_df, output_path)

    timeline = job_df.sort_values("completion").reset_index(drop=True)
    timeline["completed_jobs"] = np.arange(1, len(timeline) + 1)

    plt.figure(figsize=(12, 5))
    sns.lineplot(data=timeline, x="completion", y="completed_jobs", marker="o")
    plt.fill_between(
        timeline["completion"],
        timeline["completed_jobs"],
        alpha=0.2,
        color="#2E86AB",
    )
    plt.xlabel("Time")
    plt.ylabel("Cumulative completed jobs")
    plt.title("Throughput Trajectory – Adaptive MCP-RMS")
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logging.getLogger(__name__).info("Saved throughput trajectory -> %s", output_path)
    return output_path


def plot_performance_radar(metrics: Dict[str, Any], output_path: Path) -> Optional[Path]:
    if not metrics:
        return None

    # Construct qualitative scores (higher = better)
    makespan = metrics.get("makespan", 0.0)
    tardiness = metrics.get("avg_tardiness", 0.0)
    utilization = metrics.get("utilization", 0.0)
    energy = metrics.get("energy_consumption", 0.0)
    reconfigs = metrics.get("reconfigurations", 0.0)

    scores = {
        "Makespan": 1 / (1 + makespan / max(makespan, 1.0)),
        "Punctualité": 1 / (1 + tardiness),
        "Utilisation": np.clip(utilization, 0, 1),
        "Énergie": 1 / (1 + energy / max(energy, 1.0)),
        "Agilité": 1 / (1 + reconfigs),
    }

    categories = list(scores.keys())
    values = list(scores.values())
    values.append(values[0])  # close the loop
    categories.append(categories[0])

    fig = go.Figure(
        data=
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name="Adaptive MCP",
            line=dict(color="#FF6F61", width=3),
            fillcolor="rgba(255, 111, 97, 0.3)",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Adaptive MCP Performance Profile",
        showlegend=False,
    )
    pio.write_html(fig, output_path, include_plotlyjs="cdn", auto_open=False)
    logging.getLogger(__name__).info("Saved interactive radar -> %s", output_path)
    return output_path


def plot_interactive_timeline(schedule_df: pd.DataFrame, output_path: Path) -> Optional[Path]:
    if schedule_df.empty:
        return None

    ops_df = schedule_df[schedule_df["event"] == "operation_start"].copy()
    if ops_df.empty:
        return None

    ops_df["finish"] = ops_df["completion_time"]
    ops_df["start"] = ops_df["time"]
    ops_df["operation"] = ops_df["operation_label"]

    fig = px.timeline(
        ops_df,
        x_start="start",
        x_end="finish",
        y="machine_label",
        color="job_label",
        hover_data=["operation", "duration"],
        title="Adaptive MCP-RMS Timeline",
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(legend_title="Job", template="plotly_dark")

    pio.write_html(fig, output_path, include_plotlyjs="cdn", auto_open=False)
    logging.getLogger(__name__).info("Saved interactive timeline -> %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Solver orchestration
# ---------------------------------------------------------------------------


def run_adaptive_mcp(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    logger = logging.getLogger("monolith")

    env = RMSEnvironment(
        num_machines=args.num_machines,
        num_configs_per_machine=args.num_configs,
        seed=args.seed,
    )
    env.generate_jobs(
        num_jobs=args.num_jobs,
        min_ops=args.min_operations,
        max_ops=args.max_operations,
    )
    env.reset()

    memory_dir = output_dir / "memory_cache"
    memory = MemorySystem(
        max_episodes=args.max_memory,
        embedding_dim=args.embedding_dim,
        save_dir=str(memory_dir),
    )

    if args.disable_llm or LLMClient is None:
        llm = DisabledLLMClient("LLM disabled by command-line flag" if args.disable_llm else "LLM client unavailable")
        logger.warning("Running in heuristic-only mode – no LLM queries will be made")
    else:
        llm = LLMClient(
            base_url=args.llm_base_url,
            model=args.llm_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            request_timeout=args.llm_timeout,
            num_ctx=args.num_ctx,
            auto_select_model=not args.force_model,
        )

        if not args.force_model:
            available = llm.list_available_models(force_refresh=True)
            if available:
                formatted = ", ".join(
                    f"{m['name']} ({m.get('size', 0)/1024**3:.1f} GB)"
                    for m in available
                    if ":cloud" not in m.get("name", "")
                )
                logger.info("Local Ollama models: %s", formatted or "none")
            else:
                logger.warning(
                    "No Ollama models detected – the solver will lean on heuristics unless a model is pulled"
                )

    server = MCPServer(env=env, memory=memory, llm=llm)
    if args.disable_llm:
        server.llm_enabled = False

    logger.info(
        "Launching Adaptive MCP-RMS | machines=%d configs=%d jobs=%d iterations=%d",
        args.num_machines,
        args.num_configs,
        args.num_jobs,
        args.max_iterations,
    )

    result = server.solve(max_iterations=args.max_iterations, objective="minimize_makespan")

    metrics = result.get("final_metrics", env.compute_metrics())
    logger.info("Run completed – makespan=%.2f | tardiness=%.2f | utilization=%.1f%%",
                metrics.get("makespan", 0.0),
                metrics.get("avg_tardiness", 0.0),
                metrics.get("utilization", 0.0) * 100)

    return {
        "environment": env,
        "memory": memory,
        "llm": llm,
        "server": server,
        "result": result,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Reporting utilities
# ---------------------------------------------------------------------------


def write_summary_report(
    output_dir: Path,
    metrics: Dict[str, Any],
    stats: Dict[str, Any],
    job_df: pd.DataFrame,
    machine_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
) -> Path:
    report_path = output_dir / "summary.md"
    job_delays = job_df["tardiness"].sum() if not job_df.empty else 0.0

    llm_stats = stats.get("server", {}).get("llm", {})
    llm_status = "disabled" if isinstance(llm_stats, dict) and llm_stats.get("disabled") else "active"
    jobs_processed = len(job_df)
    max_tardiness = (
        float(job_df["tardiness"].max(skipna=True))
        if not job_df.empty
        else 0.0
    )

    lines = [
        "# Adaptive MCP-RMS Monolithic Run",
        "",
        "## KPI Snapshot",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Makespan | {metrics.get('makespan', 0.0):.2f} |",
        f"| Average tardiness | {metrics.get('avg_tardiness', 0.0):.2f} |",
        f"| Total tardiness | {metrics.get('total_tardiness', 0.0):.2f} |",
        f"| Total flowtime | {metrics.get('total_flowtime', 0.0):.2f} |",
        f"| Utilisation | {metrics.get('utilization', 0.0)*100:.1f}% |",
        f"| Energy consumption | {metrics.get('energy_consumption', 0.0):.2f} |",
        f"| Reconfigurations | {metrics.get('reconfigurations', 0)} |",
        f"| Completed jobs | {metrics.get('completed_jobs', 0)} |",
        "",
        "## Highlights",
        "",
        textwrap.dedent(
            f"""
            * **Jobs processed:** {jobs_processed}
            * **Cumulative tardiness:** {job_delays:.2f}
            * **Max tardiness:** {max_tardiness:.2f}
            * **LLM status:** {llm_status}
            * **Episodes recorded:** {stats.get('memory', {}).get('total_episodes', 0)}
            """
        ).strip(),
        "",
        "## Artefacts",
        "",
        "- `schedule_events.csv` – chronological log of operations and reconfigurations",
        "- `machine_vitals_heatmap.(png|html)` – utilisation / energy / reconfiguration map",
        "- `gantt_schedule.(png|html)` – cinematic view of the plan",
        "- `timeline.html` – interactive timeline rendered with Plotly",
        "- `performance_radar.html` – normalised KPI radar",
        "- `job_throughput.(png|html)` – throughput evolution",
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logging.getLogger(__name__).info("Wrote summary report -> %s", report_path)
    return report_path


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monolithic Adaptive MCP-RMS runner")

    parser.add_argument("--output-dir", type=Path, default=Path("monolith_output"), help="Directory where artefacts will be stored")
    parser.add_argument("--num-machines", type=int, default=5, help="Number of machines in the RMS")
    parser.add_argument("--num-configs", type=int, default=3, help="Configurations per machine")
    parser.add_argument("--num-jobs", type=int, default=20, help="Number of jobs to generate")
    parser.add_argument("--min-operations", type=int, default=3, help="Minimum operations per job")
    parser.add_argument("--max-operations", type=int, default=6, help="Maximum operations per job")
    parser.add_argument("--max-iterations", type=int, default=40, help="Adaptive MCP iteration cap")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--disable-llm", action="store_true", help="Skip all LLM calls and rely solely on heuristics")
    parser.add_argument("--llm-base-url", type=str, default="http://localhost:11434", help="Base URL for the Ollama server")
    parser.add_argument("--llm-model", type=str, default="auto", help="Preferred Ollama model (use 'auto' for lightest)")
    parser.add_argument("--temperature", type=float, default=0.6, help="LLM sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=320, help="Maximum tokens to generate per request")
    parser.add_argument("--num-ctx", type=int, default=1536, help="Context window for the local model")
    parser.add_argument("--llm-timeout", type=int, default=60, help="Timeout for Ollama requests (seconds)")
    parser.add_argument("--force-model", action="store_true", help="Do not auto-select the lightest model even after memory errors")

    parser.add_argument("--max-memory", type=int, default=2000, help="Maximum episodes to store in memory")
    parser.add_argument("--embedding-dim", type=int, default=64, help="State embedding dimensionality")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose console logging")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "monolith.log"
    configure_logging(log_file, verbose=args.verbose)

    artefacts: Dict[str, Any] = run_adaptive_mcp(args, output_dir)

    env: RMSEnvironment = artefacts["environment"]
    result: Dict[str, Any] = artefacts["result"]
    metrics: Dict[str, Any] = artefacts["metrics"]
    server: MCPServer = artefacts["server"]
    memory: MemorySystem = artefacts["memory"]

    schedule_df = build_schedule_dataframe(env.schedule)
    machine_df = extract_machine_stats(env)
    job_df = extract_job_stats(env)

    schedule_csv = output_dir / "schedule_events.csv"
    schedule_df.to_csv(schedule_csv, index=False)
    logging.getLogger(__name__).info("Saved schedule events -> %s", schedule_csv)

    job_csv = output_dir / "job_stats.csv"
    job_df.to_csv(job_csv, index=False)
    logging.getLogger(__name__).info("Saved job statistics -> %s", job_csv)

    machine_csv = output_dir / "machine_stats.csv"
    machine_df.to_csv(machine_csv, index=False)
    logging.getLogger(__name__).info("Saved machine statistics -> %s", machine_csv)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    server_stats = server.get_statistics()
    stats = {
        "server": server_stats,
        "memory": memory.get_statistics(),
        "llm": server_stats.get("llm", {}),
    }
    stats_path = output_dir / "run_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    plot_gantt(schedule_df, output_dir / "gantt_schedule.png")
    plot_machine_heatmap(machine_df, output_dir / "machine_vitals_heatmap.png")
    plot_job_completion(job_df, output_dir / "job_throughput.png")
    plot_interactive_timeline(schedule_df, output_dir / "timeline.html")
    plot_performance_radar(metrics, output_dir / "performance_radar.html")

    write_summary_report(output_dir, metrics, stats, job_df, machine_df, schedule_df)

    logging.getLogger(__name__).info("Monolithic Adaptive MCP-RMS run complete. Artefacts stored in %s", output_dir)


if __name__ == "__main__":
    main()
