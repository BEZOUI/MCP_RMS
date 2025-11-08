"""Results analysis and visualization utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


logger = logging.getLogger(__name__)


@dataclass
class MetricInfo:
    """Metadata describing a metric used in the analysis."""

    label: str
    higher_is_better: bool


class ResultsAnalyzer:
    """Analyze and visualize experimental results.

    The analyzer generates an extensive set of visualisations (30+ figures)
    covering different aspects of the benchmark results. All plots are stored
    on disk, and summary tables/statistical comparisons are produced for use
    in reports or publications.
    """

    def __init__(self, results: List[Dict], output_dir: Path | str = "data/results/figures"):
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 12

        self.metric_info: Dict[str, MetricInfo] = {
            "makespan_mean": MetricInfo("Makespan", False),
            "tardiness_mean": MetricInfo("Total Tardiness", False),
            "energy_mean": MetricInfo("Energy Consumption", False),
            "time_mean": MetricInfo("Runtime (s)", False),
            "success_rate": MetricInfo("Success Rate", True),
        }

        self.df = self._create_dataframe()
        self.generated_figures: List[Path] = []

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert nested results into a tabular DataFrame."""

        rows: List[Dict] = []

        for result in self.results:
            instance_name = result["instance_name"]
            num_machines = result["num_machines"]
            num_jobs = result["num_jobs"]

            for method, data in result["methods"].items():
                if data.get("success_rate", 0) <= 0:
                    continue

                rows.append(
                    {
                        "instance": instance_name,
                        "instance_size": f"{num_machines}x{num_jobs}",
                        "num_machines": num_machines,
                        "num_jobs": num_jobs,
                        "problem_size": num_machines * num_jobs,
                        "method": method,
                        "makespan_mean": data["makespan_mean"],
                        "makespan_std": data.get("makespan_std", np.nan),
                        "makespan_min": data.get("makespan_min", np.nan),
                        "makespan_max": data.get("makespan_max", np.nan),
                        "tardiness_mean": data.get("tardiness_mean", np.nan),
                        "energy_mean": data.get("energy_mean", np.nan),
                        "time_mean": data.get("time_mean", np.nan),
                        "success_rate": data.get("success_rate", 0.0),
                    }
                )

        df = pd.DataFrame(rows)
        if df.empty:
            logger.warning("Results DataFrame is empty – no plots will be generated.")
        return df

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------
    def _save_current_fig(self, fig: plt.Figure, filename: str) -> Path:
        """Save matplotlib figure and keep track of the output path."""

        filepath = self.output_dir / f"{filename}.png"
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        self.generated_figures.append(filepath)
        logger.info("Saved figure %s", filepath)
        return filepath

    def _has_data(self, metric: str) -> bool:
        return not self.df.empty and metric in self.df.columns and self.df[metric].notna().any()

    # ------------------------------------------------------------------
    # Plot families (30+ figures produced programmatically)
    # ------------------------------------------------------------------
    def plot_metric_overview(self, metric: str) -> Path | None:
        if not self._has_data(metric):
            return None

        info = self.metric_info[metric]
        agg = self.df.groupby("method")[metric].agg(["mean", "std"]).sort_values("mean", ascending=not info.higher_is_better)

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.bar(
            x=np.arange(len(agg)),
            height=agg["mean"],
            yerr=agg["std"],
            capsize=5,
            color="steelblue",
            alpha=0.8,
        )
        ax.set_xticks(np.arange(len(agg)))
        ax.set_xticklabels(agg.index, rotation=45, ha="right")
        ax.set_ylabel(info.label)
        ax.set_title(f"Average {info.label} by Method")
        ax.grid(axis="y", alpha=0.3)

        return self._save_current_fig(fig, f"metric_overview_{metric}")

    def plot_metric_box(self, metric: str) -> Path | None:
        if not self._has_data(metric):
            return None

        info = self.metric_info[metric]
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.boxplot(data=self.df, x="method", y=metric, ax=ax)
        ax.set_xlabel("Method")
        ax.set_ylabel(info.label)
        ax.set_title(f"Distribution of {info.label} across Methods")
        ax.tick_params(axis="x", rotation=45)

        return self._save_current_fig(fig, f"metric_boxplot_{metric}")

    def plot_metric_violin(self, metric: str) -> Path | None:
        if not self._has_data(metric):
            return None

        info = self.metric_info[metric]
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.violinplot(data=self.df, x="method", y=metric, inner="quart", ax=ax)
        ax.set_xlabel("Method")
        ax.set_ylabel(info.label)
        ax.set_title(f"Violin Plot of {info.label}")
        ax.tick_params(axis="x", rotation=45)

        return self._save_current_fig(fig, f"metric_violin_{metric}")

    def plot_metric_trend(self, metric: str, dimension: str) -> Path | None:
        if not self._has_data(metric):
            return None

        info = self.metric_info[metric]
        fig, ax = plt.subplots(figsize=(12, 7))

        for method, group in self.df.groupby("method"):
            trend = group.groupby(dimension)[metric].mean().sort_index()
            if len(trend) <= 1:
                continue
            ax.plot(trend.index, trend.values, marker="o", label=method)

        ax.set_xlabel(dimension.replace("_", " ").title())
        ax.set_ylabel(info.label)
        ax.set_title(f"{info.label} vs {dimension.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        suffix = dimension.replace("_", "")
        return self._save_current_fig(fig, f"metric_trend_{metric}_{suffix}")

    def plot_metric_heatmap(self, metric: str) -> Path | None:
        if not self._has_data(metric):
            return None

        info = self.metric_info[metric]
        pivot = self.df.pivot_table(index="method", columns="instance_size", values=metric, aggfunc="mean")
        if pivot.empty:
            return None

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax)
        ax.set_xlabel("Instance Size")
        ax.set_ylabel("Method")
        ax.set_title(f"Average {info.label} by Instance Size")

        return self._save_current_fig(fig, f"metric_heatmap_{metric}")

    # ------------------------------------------------------------------
    # Additional visualisations
    # ------------------------------------------------------------------
    def plot_performance_profiles(self) -> Path | None:
        if self.df.empty:
            return None

        fig, ax = plt.subplots(figsize=(12, 8))
        methods = self.df["method"].unique()
        instances = self.df["instance"].unique()

        for method in methods:
            ratios: List[float] = []
            for instance in instances:
                instance_data = self.df[self.df["instance"] == instance]
                if instance_data.empty:
                    continue

                best = instance_data["makespan_mean"].min()
                best = max(best, 1e-6)

                method_data = instance_data[instance_data["method"] == method]
                if method_data.empty:
                    continue

                ratio = method_data["makespan_mean"].values[0] / best
                ratios.append(ratio)

            if not ratios:
                continue

            ratios = sorted(ratios)
            cumulative = np.arange(1, len(ratios) + 1) / len(ratios)
            ax.plot(ratios, cumulative, marker="o", linewidth=2, label=method)

        ax.set_xlabel("Performance Ratio (τ)")
        ax.set_ylabel("P(ratio ≤ τ)")
        ax.set_title("Performance Profiles (Makespan)")
        ax.set_xlim(left=1.0)
        ax.grid(True, alpha=0.3)
        ax.legend()

        return self._save_current_fig(fig, "performance_profiles")

    def plot_computational_time(self) -> Path | None:
        if not self._has_data("time_mean"):
            return None

        time_data = self.df.groupby("method")["time_mean"].agg(["mean", "std"]).sort_values("mean")
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.bar(np.arange(len(time_data)), time_data["mean"], yerr=time_data["std"], capsize=5, color="coral")
        ax.set_xticks(np.arange(len(time_data)))
        ax.set_xticklabels(time_data.index, rotation=45, ha="right")
        ax.set_ylabel("Runtime (s)")
        ax.set_title("Average Computational Time by Method")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)

        return self._save_current_fig(fig, "computational_time")

    def plot_scalability(self) -> Path | None:
        if self.df.empty:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        metrics = [
            ("makespan_mean", "Makespan"),
            ("time_mean", "Runtime (s)"),
            ("tardiness_mean", "Total Tardiness"),
            ("energy_mean", "Energy Consumption"),
        ]

        for ax, (metric, label) in zip(axes.flat, metrics):
            positive_only = True
            for method, group in self.df.groupby("method"):
                trend = group.groupby("problem_size")[metric].mean().sort_index()
                if len(trend) <= 1:
                    continue
                ax.plot(trend.index, trend.values, marker="o", label=method)
                if (trend.values <= 0).any():
                    positive_only = False

            ax.set_xlabel("Problem Size (machines × jobs)")
            ax.set_ylabel(label)
            ax.set_title(f"{label} vs Problem Size")
            ax.grid(True, alpha=0.3)
            if ax in axes[:, 1] and positive_only:
                ax.set_yscale("log")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        return self._save_current_fig(fig, "scalability_analysis")

    def plot_improvement_heatmap(self, baseline: str = "fifo") -> Path | None:
        if self.df.empty or baseline not in self.df["method"].unique():
            return None

        instances = self.df["instance"].unique()
        methods = [m for m in self.df["method"].unique() if m != baseline]
        improvements = np.zeros((len(instances), len(methods)))

        for i, instance in enumerate(instances):
            instance_data = self.df[self.df["instance"] == instance]
            baseline_data = instance_data[instance_data["method"] == baseline]
            if baseline_data.empty:
                continue
            baseline_makespan = baseline_data["makespan_mean"].values[0]

            for j, method in enumerate(methods):
                method_data = instance_data[instance_data["method"] == method]
                if method_data.empty:
                    continue
                method_makespan = method_data["makespan_mean"].values[0]
                improvements[i, j] = ((baseline_makespan - method_makespan) / baseline_makespan) * 100

        fig, ax = plt.subplots(figsize=(14, 8))
        heatmap = ax.imshow(improvements, cmap="RdYlGn", aspect="auto", vmin=-25, vmax=25)
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(instances)))
        ax.set_yticklabels(instances)
        ax.set_title(f"Improvement over {baseline.upper()} (%)")

        for i in range(len(instances)):
            for j in range(len(methods)):
                ax.text(j, i, f"{improvements[i, j]:.1f}", ha="center", va="center", color="black", fontsize=8)

        cbar = plt.colorbar(heatmap, ax=ax)
        cbar.set_label("Improvement (%)", rotation=270, labelpad=20)

        fig.tight_layout()
        return self._save_current_fig(fig, f"improvement_heatmap_{baseline}")

    # ------------------------------------------------------------------
    # Statistical analysis utilities
    # ------------------------------------------------------------------
    def statistical_analysis(self, save: bool = True) -> pd.DataFrame:
        if self.df.empty:
            return pd.DataFrame()

        methods = sorted(self.df["method"].unique())
        instances = self.df["instance"].unique()
        results: List[Dict] = []

        for i, method1 in enumerate(methods):
            for method2 in methods[i + 1 :]:
                data1: List[float] = []
                data2: List[float] = []

                for instance in instances:
                    instance_data = self.df[self.df["instance"] == instance]
                    m1 = instance_data[instance_data["method"] == method1]
                    m2 = instance_data[instance_data["method"] == method2]

                    if m1.empty or m2.empty:
                        continue

                    data1.append(m1["makespan_mean"].values[0])
                    data2.append(m2["makespan_mean"].values[0])

                if len(data1) < 3:
                    continue

                diff = np.array(data1) - np.array(data2)
                statistic, pvalue = stats.ttest_rel(data1, data2)
                try:
                    w_stat, w_pvalue = stats.wilcoxon(data1, data2)
                except ValueError:
                    w_stat, w_pvalue = np.nan, np.nan

                cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-9)

                results.append(
                    {
                        "method1": method1,
                        "method2": method2,
                        "mean_diff": float(np.mean(diff)),
                        "t_statistic": float(statistic),
                        "t_pvalue": float(pvalue),
                        "w_statistic": float(w_stat),
                        "w_pvalue": float(w_pvalue),
                        "cohens_d": float(cohens_d),
                        "significant_t": bool(pvalue < 0.05),
                        "significant_w": bool(w_pvalue < 0.05) if not np.isnan(w_pvalue) else False,
                    }
                )

        stats_df = pd.DataFrame(results)
        if save and not stats_df.empty:
            stats_df.to_csv(self.output_dir / "statistical_analysis.csv", index=False)
            logger.info("Saved statistical analysis to %s", self.output_dir / "statistical_analysis.csv")

        return stats_df

    def generate_pairwise_comparisons(self, top_n: int = 10, save: bool = True) -> pd.DataFrame:
        stats_df = self.statistical_analysis(save=False)
        if stats_df.empty:
            return stats_df

        stats_df["abs_mean_diff"] = stats_df["mean_diff"].abs()
        top = stats_df.sort_values("abs_mean_diff", ascending=False).head(top_n)

        if save:
            top.to_csv(self.output_dir / "top_pairwise_comparisons.csv", index=False)
            logger.info("Saved top %d pairwise comparisons", len(top))

        return top

    def create_comparison_table(self, save: bool = True) -> pd.DataFrame:
        if self.df.empty:
            return pd.DataFrame()

        comparison = (
            self.df.groupby("method").agg(
                makespan_mean_mean=("makespan_mean", "mean"),
                makespan_mean_std=("makespan_mean", "std"),
                tardiness_mean_mean=("tardiness_mean", "mean"),
                tardiness_mean_std=("tardiness_mean", "std"),
                energy_mean_mean=("energy_mean", "mean"),
                energy_mean_std=("energy_mean", "std"),
                time_mean_mean=("time_mean", "mean"),
                time_mean_std=("time_mean", "std"),
                success_rate_mean=("success_rate", "mean"),
            )
        ).round(3)

        if save:
            comparison.to_csv(self.output_dir / "comparison_table.csv")
            logger.info("Saved comparison table to %s", self.output_dir / "comparison_table.csv")

        return comparison

    def compare_to_reference_algorithms(
        self,
        reference_methods: List[str] | None = None,
        primary_method: str = "adaptive_mcp_rms",
        metrics: List[str] | None = None,
        save: bool = True,
    ) -> pd.DataFrame:
        """Create a focused comparison against key literature baselines.

        The default configuration matches the three algorithms highlighted in
        Andersen et al. (2024): a genetic algorithm, simulated annealing, and a
        reinforcement learning agent. The method summarises absolute and
        relative differences between the primary approach and each reference
        algorithm, enabling direct reporting of the study's performance against
        these canonical RMS optimisation techniques.
        """

        if self.df.empty:
            return pd.DataFrame()

        if reference_methods is None:
            reference_methods = [
                "genetic_algorithm",
                "simulated_annealing",
                "simple_dqn",
            ]

        if metrics is None:
            metrics = [
                "makespan_mean",
                "tardiness_mean",
                "time_mean",
                "success_rate",
            ]

        available_methods = set(self.df["method"].unique())

        if primary_method not in available_methods:
            logger.warning("Primary method %s not present in results", primary_method)
            return pd.DataFrame()

        primary_df = self.df[self.df["method"] == primary_method]

        rows: List[Dict[str, float]] = []

        for method in reference_methods:
            if method not in available_methods:
                logger.info("Skipping reference method %s – no results available", method)
                continue

            ref_df = self.df[self.df["method"] == method]
            row: Dict[str, float | str] = {
                "primary_method": primary_method,
                "reference_method": method,
            }

            for metric in metrics:
                if metric not in self.df.columns or (
                    primary_df[metric].dropna().empty or ref_df[metric].dropna().empty
                ):
                    row[f"{metric}_primary"] = np.nan
                    row[f"{metric}_reference"] = np.nan
                    row[f"{metric}_diff"] = np.nan
                    row[f"{metric}_rel_improvement_pct"] = np.nan
                    continue

                primary_val = float(primary_df[metric].mean())
                reference_val = float(ref_df[metric].mean())

                row[f"{metric}_primary"] = round(primary_val, 3)
                row[f"{metric}_reference"] = round(reference_val, 3)
                row[f"{metric}_diff"] = round(reference_val - primary_val, 3)

                if abs(reference_val) < 1e-9:
                    rel_improvement = np.nan
                else:
                    rel_improvement = 100.0 * (reference_val - primary_val) / abs(reference_val)

                row[f"{metric}_rel_improvement_pct"] = (
                    round(rel_improvement, 2) if not np.isnan(rel_improvement) else np.nan
                )

            rows.append(row)

        comparison_df = pd.DataFrame(rows)

        if save and not comparison_df.empty:
            output_path = self.output_dir / "reference_algorithm_comparison.csv"
            comparison_df.to_csv(output_path, index=False)
            logger.info("Saved reference algorithm comparison to %s", output_path)

        return comparison_df

    # ------------------------------------------------------------------
    # Report orchestrator
    # ------------------------------------------------------------------
    def generate_report(self, baseline: str = "fifo") -> Dict[str, pd.DataFrame]:
        if self.df.empty:
            logger.warning("No results available to analyze.")
            return {}

        logger.info("Generating analysis report with %d methods", self.df["method"].nunique())
        self.generated_figures.clear()

        for metric in self.metric_info.keys():
            self.plot_metric_overview(metric)
            self.plot_metric_box(metric)
            self.plot_metric_violin(metric)
            self.plot_metric_trend(metric, "num_jobs")
            self.plot_metric_trend(metric, "num_machines")
            self.plot_metric_heatmap(metric)

        # Additional figures beyond the 30 core charts
        self.plot_performance_profiles()
        self.plot_computational_time()
        self.plot_scalability()
        self.plot_improvement_heatmap(baseline=baseline)

        logger.info("Generated %d figures", len(self.generated_figures))

        comparison_table = self.create_comparison_table(save=True)
        stats_table = self.statistical_analysis(save=True)
        top_comparisons = self.generate_pairwise_comparisons(top_n=10, save=True)
        reference_comparison = self.compare_to_reference_algorithms(save=True)

        return {
            "comparison_table": comparison_table,
            "statistics": stats_table,
            "top_comparisons": top_comparisons,
            "reference_comparison": reference_comparison,
        }
