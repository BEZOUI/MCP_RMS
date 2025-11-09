#!/bin/bash

# Run Adaptive MCP-RMS Experiments
# Usage: ./run_experiments.sh [mode]

echo "======================================"
echo "Adaptive MCP-RMS Experimental Framework"
echo "======================================"

# Set environment variables (override as needed)
# export OLLAMA_BASE_URL="http://localhost:11434"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Default mode
MODE=${1:-full}

echo "Mode: $MODE"
echo ""

# Activate virtual environment if exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

# Create directories
mkdir -p data/instances
mkdir -p data/results
mkdir -p data/memory
mkdir -p config

# Run experiments based on mode
case $MODE in
    "quick-test")
        echo "Running quick test..."
        python main.py --quick-test
        ;;
    
    "generate")
        echo "Generating benchmark instances..."
        python main.py --mode generate --suite benchmark_suite
        ;;
    
    "run")
        echo "Running experiments..."
        python main.py --mode run --suite benchmark_suite --trials 10
        ;;
    
    "analyze")
        echo "Analyzing results..."
        python main.py --mode analyze --suite benchmark_suite
        ;;
    
    "full")
        echo "Running full experimental pipeline..."
        python main.py --mode full --suite benchmark_suite --trials 10
        ;;
    
    "small")
        echo "Running small-scale experiments..."
        python main.py --mode full --suite small_suite --trials 5 \
            --methods fifo spt edd adaptive_mcp_rms
        ;;
    
    *)
        echo "Unknown mode: $MODE"
        echo "Available modes: quick-test, generate, run, analyze, full, small"
        exit 1
        ;;
esac

echo ""
echo "======================================"
echo "Execution completed!"
echo "======================================"