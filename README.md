# Adaptive MCP-RMS: Adaptive Model Context Protocol for Dynamic Reconfigurable Manufacturing Systems

Complete implementation of the Adaptive MCP-RMS framework for intelligent manufacturing optimization using Large Language Models.

## Features

- **Complete RMS Simulation**: Discrete-event simulation of reconfigurable manufacturing systems
- **Self-Learning Primitives**: Operations with episodic memory for experience-based learning
- **LLM Integration**: Claude API integration for intelligent reasoning
- **Comprehensive Baselines**: FIFO, SPT, EDD, MWKR, GA, SA implementations
- **Full Experimental Pipeline**: Benchmark generation, execution, and analysis
- **Rich Visualizations**: Performance profiles, scalability analysis, statistical comparisons

## Installation
```bash
# Clone repository
git clone <repository-url>
cd adaptive_mcp_rms

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Quick Start

### Quick Test
```bash
python main.py --quick-test
```

### Full Experimental Pipeline
```bash
# Generate benchmarks, run experiments, and analyze
python main.py --mode full --trials 10

# Or use the shell script
chmod +x run_experiments.sh
./run_experiments.sh full
```

### Individual Phases
```bash
# Generate benchmark instances
python main.py --mode generate --suite benchmark_suite

# Run experiments
python main.py --mode run --suite benchmark_suite --trials 10

# Analyze results
python main.py --mode analyze --suite benchmark_suite
```

### Custom Experiments
```bash
# Run specific methods
python main.py --mode run --methods fifo spt adaptive_mcp_rms --trials 5

# Small-scale testing
./run_experiments.sh small
```

## Project Structure
```
adaptive_mcp_rms/
├── src/
│   ├── environment.py          # RMS simulation
│   ├── primitives.py           # MCP primitives
│   ├── memory.py               # Self-learning memory
│   ├── mcp_server.py           # MCP server
│   ├── llm_client.py           # LLM client
│   ├── baselines.py            # Comparison methods
│   └── utils.py                # Utilities
├── experiments/
│   ├── benchmark_generator.py  # Instance generation
│   ├── experiment_runner.py    # Experiment execution
│   └── analysis.py             # Results analysis
├── data/
│   ├── instances/              # Benchmark instances
│   ├── results/                # Experimental results
│   └── memory/                 # Memory storage
├── config/
│   └── config.yaml             # Configuration
├── main.py                     # Main entry point
├── run_experiments.sh          # Bash script
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Configuration

Edit `config/config.yaml` to customize:

- Environment parameters (machines, jobs)
- LLM settings (model, temperature)
- Algorithm parameters (GA, SA)
- Experiment settings (trials, methods)

## Methods Implemented

1. **Adaptive MCP-RMS**: Our novel approach with self-learning primitives
2. **FIFO**: First In First Out
3. **SPT**: Shortest Processing Time
4. **EDD**: Earliest Due Date
5. **MWKR**: Most Work Remaining
6. **Genetic Algorithm**: Population-based metaheuristic
7. **Simulated Annealing**: Temperature-based local search

## Results and Analysis

Results are saved to `data/results/`:
- `summary_table.csv`: Comprehensive comparison
- `statistical_analysis.csv`: Statistical tests
- `figures/`: All generated plots

Plots include:
- Makespan comparisons
- Performance profiles
- Scalability analysis
- Computational time
- Improvement heatmaps

## Example Usage
```python
from src.environment import RMSEnvironment
from src.memory import MemorySystem
from src.llm_client import LLMClient
from src.mcp_server import MCPServer

# Create environment
env = RMSEnvironment(num_machines=10, seed=42)
env.generate_jobs(num_jobs=30)

# Initialize components
memory = MemorySystem()
llm = LLMClient(api_key="your-key")
server = MCPServer(env, memory, llm)

# Solve
result = server.solve(max_iterations=50)
print(f"Makespan: {result['final_metrics']['makespan']:.2f}")
```

## Citation

If you use this code, please cite:
```bibtex
@article{bezoui2024adaptive,
  title={Adaptive Model Context Protocol for Dynamic Reconfigurable Manufacturing Systems with Self-Learning Primitives},
  author={Bezoui, Madani and Bounceur, Ahcene},
  journal={IEEE Conference on ...},
  year={2024}
}
```

## License

MIT License

## Contact

For questions or issues, please contact: mbezoui@cesi.fr