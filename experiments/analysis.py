"""
Results analysis and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyze and visualize experimental results"""
    
    def __init__(self, results: List[Dict], output_dir: str = "data/results/figures"):
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
        # Create DataFrame
        self.df = self._create_dataframe()
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame"""
        rows = []
        
        for result in self.results:
            instance_name = result['instance_name']
            num_machines = result['num_machines']
            num_jobs = result['num_jobs']
            
            for method, data in result['methods'].items():
                if data.get('success_rate', 0) > 0:
                    row = {
                        'instance': instance_name,
                        'instance_size': f"{num_machines}x{num_jobs}",
                        'num_machines': num_machines,
                        'num_jobs': num_jobs,
                        'method': method,
                        'makespan_mean': data['makespan_mean'],
                        'makespan_std': data['makespan_std'],
                        'makespan_min': data['makespan_min'],
                        'makespan_max': data['makespan_max'],
                        'tardiness_mean': data['tardiness_mean'],
                        'energy_mean': data['energy_mean'],
                        'time_mean': data['time_mean'],
                        'success_rate': data['success_rate']
                    }
                    rows.append(row)
        
        return pd.DataFrame(rows)
    
    def plot_makespan_comparison(self, save: bool = True):
        """Plot makespan comparison across methods"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Group by instance size
        small = self.df[self.df['instance'].str.contains('small')]
        medium = self.df[self.df['instance'].str.contains('medium')]
        large = self.df[self.df['instance'].str.contains('large')]
        
        datasets = [
            (small, 'Small Instances (5x20)', axes[0]),
            (medium, 'Medium Instances (15x50)', axes[1]),
            (large, 'Large Instances (30x100)', axes[2])
        ]
        
        for data, title, ax in datasets:
            if len(data) > 0:
                # Group by method and calculate mean
                grouped = data.groupby('method')['makespan_mean'].agg(['mean', 'std'])
                grouped = grouped.sort_values('mean')
                
                # Plot
                x = np.arange(len(grouped))
                ax.bar(x, grouped['mean'], yerr=grouped['std'], 
                      capsize=5, alpha=0.7, color='steelblue')
                ax.set_xticks(x)
                ax.set_xticklabels(grouped.index, rotation=45, ha='right')
                ax.set_ylabel('Makespan')
                ax.set_title(title)
                ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'makespan_comparison.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved makespan comparison plot")
        
        plt.show()
    
    def plot_performance_profiles(self, save: bool = True):
        """Plot performance profiles (cumulative distribution)"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get unique methods
        methods = self.df['method'].unique()
        
        # Calculate performance ratios for each instance
        instances = self.df['instance'].unique()
        
        for method in methods:
            ratios = []
            
            for instance in instances:
                instance_data = self.df[self.df['instance'] == instance]
                
                if len(instance_data) > 0:
                    # Get best makespan for this instance
                    best_makespan = instance_data['makespan_mean'].min()
                    
                    # Get this method's makespan
                    method_data = instance_data[instance_data['method'] == method]
                    
                    if len(method_data) > 0:
                        method_makespan = method_data['makespan_mean'].values[0]
                        ratio = method_makespan / best_makespan
                        ratios.append(ratio)
            
            if ratios:
                # Sort ratios
                ratios = sorted(ratios)
                
                # Calculate cumulative distribution
                n = len(ratios)
                cumulative = np.arange(1, n + 1) / n
                
                # Plot
                ax.plot(ratios, cumulative, marker='o', label=method, linewidth=2)
        
        ax.set_xlabel('Performance Ratio (τ)')
        ax.set_ylabel('P(ratio ≤ τ)')
        ax.set_title('Performance Profiles')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1.0, None)
        
        if save:
            plt.savefig(self.output_dir / 'performance_profiles.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved performance profiles plot")
        
        plt.show()
    
    def plot_computational_time(self, save: bool = True):
        """Plot computational time comparison"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group by method
        time_data = self.df.groupby('method')['time_mean'].agg(['mean', 'std'])
        time_data = time_data.sort_values('mean')
        
        # Plot
        x = np.arange(len(time_data))
        ax.bar(x, time_data['mean'], yerr=time_data['std'], 
              capsize=5, alpha=0.7, color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(time_data.index, rotation=45, ha='right')
        ax.set_ylabel('Computational Time (seconds)')
        ax.set_title('Average Computational Time by Method')
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'computational_time.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved computational time plot")
        
        plt.show()
    
    def plot_scalability(self, save: bool = True):
        """Plot scalability analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract size information
        self.df['problem_size'] = self.df['num_machines'] * self.df['num_jobs']
        
        methods = self.df['method'].unique()
        
        # Plot 1: Makespan vs Problem Size
        ax = axes[0, 0]
        for method in methods:
            method_data = self.df[self.df['method'] == method]
            grouped = method_data.groupby('problem_size')['makespan_mean'].mean()
            ax.plot(grouped.index, grouped.values, marker='o', label=method)
        ax.set_xlabel('Problem Size (machines × jobs)')
        ax.set_ylabel('Makespan')
        ax.set_title('Makespan vs Problem Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Computational Time vs Problem Size
        ax = axes[0, 1]
        for method in methods:
            method_data = self.df[self.df['method'] == method]
            grouped = method_data.groupby('problem_size')['time_mean'].mean()
            ax.plot(grouped.index, grouped.values, marker='o', label=method)
        ax.set_xlabel('Problem Size (machines × jobs)')
        ax.set_ylabel('Computational Time (s)')
        ax.set_title('Computational Time vs Problem Size')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Tardiness vs Problem Size
        ax = axes[1, 0]
        for method in methods:
            method_data = self.df[self.df['method'] == method]
            grouped = method_data.groupby('problem_size')['tardiness_mean'].mean()
            ax.plot(grouped.index, grouped.values, marker='o', label=method)
        ax.set_xlabel('Problem Size (machines × jobs)')
        ax.set_ylabel('Total Tardiness')
        ax.set_title('Tardiness vs Problem Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Energy vs Problem Size
        ax = axes[1, 1]
        for method in methods:
            method_data = self.df[self.df['method'] == method]
            grouped = method_data.groupby('problem_size')['energy_mean'].mean()
            ax.plot(grouped.index, grouped.values, marker='o', label=method)
        ax.set_xlabel('Problem Size (machines × jobs)')
        ax.set_ylabel('Energy Consumption')
        ax.set_title('Energy Consumption vs Problem Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'scalability_analysis.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved scalability analysis plot")
        
        plt.show()
    
    def plot_improvement_heatmap(self, baseline: str = 'fifo', save: bool = True):
        """Plot improvement heatmap relative to baseline"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate improvements
        instances = self.df['instance'].unique()
        methods = [m for m in self.df['method'].unique() if m != baseline]
        
        improvements = np.zeros((len(instances), len(methods)))
        
        for i, instance in enumerate(instances):
            instance_data = self.df[self.df['instance'] == instance]
            
            # Get baseline performance
            baseline_data = instance_data[instance_data['method'] == baseline]
            if len(baseline_data) == 0:
                continue
            
            baseline_makespan = baseline_data['makespan_mean'].values[0]
            
            for j, method in enumerate(methods):
                method_data = instance_data[instance_data['method'] == method]
                if len(method_data) > 0:
                    method_makespan = method_data['makespan_mean'].values[0]
                    improvement = ((baseline_makespan - method_makespan) / baseline_makespan) * 100
                    improvements[i, j] = improvement
        
        # Plot heatmap
        im = ax.imshow(improvements, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=20)
        
        # Set ticks
        ax.set_xticks(np.arange(len(methods)))
        ax.set_yticks(np.arange(len(instances)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_yticklabels(instances)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Improvement (%)', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(instances)):
            for j in range(len(methods)):
                text = ax.text(j, i, f'{improvements[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title(f'Improvement over {baseline.upper()} (%)')
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / f'improvement_heatmap_{baseline}.png', 
                       dpi=300, bbox_inches='tight')
            logger.info(f"Saved improvement heatmap")
        
        plt.show()
    
    def statistical_analysis(self, save: bool = True) -> pd.DataFrame:
        """Perform statistical analysis"""
        logger.info("Performing statistical analysis...")
        
        # Pairwise comparisons
        methods = self.df['method'].unique()
        instances = self.df['instance'].unique()
        
        results = []
        
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                # Collect paired data
                data1 = []
                data2 = []
                
                for instance in instances:
                    instance_data = self.df[self.df['instance'] == instance]
                    
                    m1_data = instance_data[instance_data['method'] == method1]
                    m2_data = instance_data[instance_data['method'] == method2]
                    
                    if len(m1_data) > 0 and len(m2_data) > 0:
                        data1.append(m1_data['makespan_mean'].values[0])
                        data2.append(m2_data['makespan_mean'].values[0])
                
                if len(data1) >= 3:  # Need at least 3 samples
                    # Paired t-test
                    statistic, pvalue = stats.ttest_rel(data1, data2)
                    
                    # Wilcoxon signed-rank test
                    w_statistic, w_pvalue = stats.wilcoxon(data1, data2)
                    
                    # Effect size (Cohen's d)
                    diff = np.array(data1) - np.array(data2)
                    cohens_d = np.mean(diff) / np.std(diff, ddof=1)
                    
                    results.append({
                        'method1': method1,
                        'method2': method2,
                        'mean_diff': np.mean(diff),
                        't_statistic': statistic,
                        't_pvalue': pvalue,
                        'w_statistic': w_statistic,
                        'w_pvalue': w_pvalue,
                        'cohens_d': cohens_d,
                        'significant_t': pvalue < 0.05,
                        'significant_w': w_pvalue < 0.05
                    })
        
        stats_df = pd.DataFrame(results)
        
        if save:
            stats_df.to_csv(self.output_dir / 'statistical_analysis.csv', index=False)
            logger.info("Saved statistical analysis")
        
        return stats_df
    
    def create_comparison_table(self, save: bool = True) -> pd.DataFrame:
        """Create comprehensive comparison table"""
        # Group by method
        comparison = self.df.groupby('method').agg({
            'makespan_mean': ['mean', 'std'],
            'tardiness_mean': ['mean', 'std'],
            'energy_mean': ['mean', 'std'],
            'time_mean': ['mean', 'std'],
            'success_rate': 'mean'
        }).round(2)
        
        # Flatten column names
        comparison.columns = ['_'.join(col).strip() for col in comparison.columns.values]