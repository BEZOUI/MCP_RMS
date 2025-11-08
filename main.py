"""
Main execution script for Adaptive MCP-RMS experiments
"""

import argparse
import logging
from pathlib import Path
import sys

from src.utils import setup_logging, load_config
from experiments.benchmark_generator import BenchmarkGenerator
from experiments.experiment_runner import ExperimentRunner
from experiments.analysis import ResultsAnalyzer


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Adaptive MCP-RMS Experiments')
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, 
                       choices=['generate', 'run', 'analyze', 'full'],
                       default='full',
                       help='Execution mode')
    parser.add_argument('--methods', nargs='+', default=None,
                       help='Methods to run (overrides config)')
    parser.add_argument('--trials', type=int, default=None,
                       help='Number of trials (overrides config)')
    parser.add_argument('--suite', type=str, default='benchmark_suite',
                       help='Benchmark suite name')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Setup logging
    setup_logging(
        log_file=config['output']['log_file'],
        level=getattr(logging, config['output']['log_level'])
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("ADAPTIVE MCP-RMS EXPERIMENTAL FRAMEWORK")
    logger.info("="*80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {args.config}")
    
    # Override config with command line args
    if args.methods:
        config['experiments']['methods'] = args.methods
    if args.trials:
        config['experiments']['num_trials'] = args.trials
    
    # Execute based on mode
    if args.mode in ['generate', 'full']:
        logger.info("\n" + "="*80)
        logger.info("PHASE 1: GENERATING BENCHMARK INSTANCES")
        logger.info("="*80)
        
        generator = BenchmarkGenerator()
        instances = generator.generate_suite(args.suite)
        
        logger.info(f"Generated {len(instances)} benchmark instances")
    
    if args.mode in ['run', 'full']:
        logger.info("\n" + "="*80)
        logger.info("PHASE 2: RUNNING EXPERIMENTS")
        logger.info("="*80)
        
        runner = ExperimentRunner(config)
        results = runner.run_benchmark_suite(
            suite_name=args.suite,
            methods=config['experiments']['methods'],
            num_trials=config['experiments']['num_trials']
        )
        
        logger.info(f"Completed {len(results)} instance experiments")
        
        # Create summary table
        summary_df = runner.create_summary_table()
        logger.info("\nSummary Table:")
        logger.info("\n" + summary_df.to_string())
    
    if args.mode in ['analyze', 'full']:
        logger.info("\n" + "="*80)
        logger.info("PHASE 3: ANALYZING RESULTS")
        logger.info("="*80)
        
        from src.utils import load_results
        
        # Load results
        results_file = Path(config['experiments']['results_dir']) / f"{args.suite}_results.json"
        if not results_file.exists():
            logger.error(f"Results file not found: {results_file}")
            sys.exit(1)
        
        results = load_results(results_file)
        
        # Analyze
        analyzer = ResultsAnalyzer(
            results,
            output_dir=Path(config['experiments']['results_dir']) / 'figures'
        )
        
        if config['output']['generate_plots']:
            analyzer.generate_report()
        
        logger.info("Analysis completed!")
    
    logger.info("\n" + "="*80)
    logger.info("EXECUTION COMPLETED SUCCESSFULLY")
    logger.info("="*80)


def run_quick_test():
    """Run a quick test with small instance"""
    from src.environment import RMSEnvironment
    from src.memory import MemorySystem
    from src.llm_client import LLMClient
    from src.mcp_server import MCPServer
    from src.baselines import DispatchingRules
    
    print("Running quick test...")
    
    # Create small environment
    env = RMSEnvironment(num_machines=5, num_configs_per_machine=3, seed=42)
    env.generate_jobs(num_jobs=10, min_ops=2, max_ops=4)
    
    print(f"\nEnvironment created:")
    print(f"  Machines: {len(env.machines)}")
    print(f"  Jobs: {len(env.jobs)}")
    
    # Test dispatching rules
    print("\n1. Testing FIFO...")
    dr = DispatchingRules(env)
    metrics = dr.fifo()
    print(f"   Makespan: {metrics['makespan']:.2f}")
    print(f"   Tardiness: {metrics['total_tardiness']:.2f}")
    
    print("\n2. Testing SPT...")
    metrics = dr.spt()
    print(f"   Makespan: {metrics['makespan']:.2f}")
    print(f"   Tardiness: {metrics['total_tardiness']:.2f}")
    
    print("\n3. Testing EDD...")
    metrics = dr.edd()
    print(f"   Makespan: {metrics['makespan']:.2f}")
    print(f"   Tardiness: {metrics['total_tardiness']:.2f}")
    
    print("\nQuick test completed successfully!")


if __name__ == "__main__":
    import os
    
    # Check if running as quick test
    if len(sys.argv) > 1 and sys.argv[1] == '--quick-test':
        run_quick_test()
    else:
        main()