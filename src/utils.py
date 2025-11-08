"""
Utility functions
"""

import numpy as np
import json
import yaml
from pathlib import Path
from typing import Dict, Any
import logging


def setup_logging(log_file: str = "rms.log", level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_results(results: Dict, filepath: str):
    """Save results to JSON file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj
    
    results_converted = convert(results)
    
    with open(filepath, 'w') as f:
        json.dump(results_converted, f, indent=2)


def load_results(filepath: str) -> Dict:
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results


def compute_improvement(baseline: float, improved: float, 
                       minimize: bool = True) -> float:
    """Compute percentage improvement"""
    if baseline == 0:
        return 0.0
    
    if minimize:
        improvement = ((baseline - improved) / baseline) * 100
    else:
        improvement = ((improved - baseline) / baseline) * 100
    
    return improvement


def statistical_test(data1: np.ndarray, data2: np.ndarray, 
                     test: str = 'ttest') -> Dict:
    """Perform statistical test"""
    from scipy import stats
    
    if test == 'ttest':
        statistic, pvalue = stats.ttest_ind(data1, data2)
    elif test == 'wilcoxon':
        statistic, pvalue = stats.wilcoxon(data1, data2)
    elif test == 'mannwhitney':
        statistic, pvalue = stats.mannwhitneyu(data1, data2)
    else:
        raise ValueError(f"Unknown test: {test}")
    
    return {
        'statistic': float(statistic),
        'pvalue': float(pvalue),
        'significant': pvalue < 0.05
    }


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.2f}m"
    else:
        return f"{seconds/3600:.2f}h"