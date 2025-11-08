#!/usr/bin/env python3
"""
Python-based experiment runner (cross-platform)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_command(cmd):
    """Run command and stream output"""
    print(f"\nExecuting: {' '.join(cmd)}")
    print("-" * 60)
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode


def setup_environment():
    """Setup virtual environment and install dependencies"""
    print("Setting up environment...")
    
    # Create directories
    for dir_path in ['data/instances', 'data/results', 'data/memory', 'config']:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Check if venv exists
    if not Path('venv').exists():
        print("Creating virtual environment...")
        run_command([sys.executable, '-m', 'venv', 'venv'])
        
        # Determine pip path
        if sys.platform == 'win32':
            pip_path = 'venv/Scripts/pip'
        else:
            pip_path = 'venv/bin/pip'
        
        print("Installing dependencies...")
        run_command([pip_path, 'install', '-r', 'requirements.txt'])


def main():
    parser = argparse.ArgumentParser(description='Run Adaptive MCP-RMS Experiments')
    parser.add_argument('mode', choices=['quick-test', 'generate', 'run', 'analyze', 'full', 'small'],
                       help='Execution mode')
    parser.add_argument('--setup', action='store_true',
                       help='Setup environment first')
    parser.add_argument('--api-key', type=str,
                       help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Setup if requested
    if args.setup:
        setup_environment()
    
    # Set API key
    if args.api_key:
        os.environ['ANTHROPIC_API_KEY'] = args.api_key
    
    # Check API key for non-test modes
    if args.mode != 'quick-test' and 'ANTHROPIC_API_KEY' not in os.environ:
        print("WARNING: ANTHROPIC_API_KEY not set!")
        print("Set it with: export ANTHROPIC_API_KEY='your-key' or use --api-key")
        if 'adaptive_mcp_rms' in ['full', 'run']:
            print("Adaptive MCP-RMS will be skipped.")
    
    print("=" * 70)
    print("ADAPTIVE MCP-RMS EXPERIMENTAL FRAMEWORK")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print()
    
    # Build command
    if args.mode == 'quick-test':
        cmd = [sys.executable, 'main.py', '--quick-test']
    elif args.mode == 'generate':
        cmd = [sys.executable, 'main.py', '--mode', 'generate', '--suite', 'benchmark_suite']
    elif args.mode == 'run':
        cmd = [sys.executable, 'main.py', '--mode', 'run', '--suite', 'benchmark_suite', '--trials', '10']
    elif args.mode == 'analyze':
        cmd = [sys.executable, 'main.py', '--mode', 'analyze', '--suite', 'benchmark_suite']
    elif args.mode == 'full':
        cmd = [sys.executable, 'main.py', '--mode', 'full', '--suite', 'benchmark_suite', '--trials', '10']
    elif args.mode == 'small':
        cmd = [sys.executable, 'main.py', '--mode', 'full', '--suite', 'small_suite', 
               '--trials', '5', '--methods', 'fifo', 'spt', 'edd', 'mwkr']
    
    # Run command
    returncode = run_command(cmd)
    
    print()
    print("=" * 70)
    if returncode == 0:
        print("EXECUTION COMPLETED SUCCESSFULLY")
    else:
        print(f"EXECUTION FAILED (exit code: {returncode})")
    print("=" * 70)
    
    return returncode


if __name__ == '__main__':
    sys.exit(main())