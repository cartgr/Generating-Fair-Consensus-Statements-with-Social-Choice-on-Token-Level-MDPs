# Generating Fair Consensus Statements with Social Choice on Token-Level MDPs

This repository contains the code and experiments for the paper "Generating Fair Consensus Statements with Social Choice on Token-Level MDPs" submitted to NeurIPS 2025.

## Table of Contents
- [Generating Fair Consensus Statements with Social Choice on Token-Level MDPs](#generating-fair-consensus-statements-with-social-choice-on-token-level-mdps)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Methods Implemented](#methods-implemented)
  - [Setup and Installation](#setup-and-installation)
  - [Running Experiments](#running-experiments)
    - [AAMAS Experiments](#aamas-experiments)
    - [Custom Experiments](#custom-experiments)
    - [Concurrent Execution](#concurrent-execution)
    - [Running Multiple Seeds](#running-multiple-seeds)
  - [Evaluation Workflow](#evaluation-workflow)
    - [Evaluation Metrics](#evaluation-metrics)
    - [Post-hoc Evaluation](#post-hoc-evaluation)
  - [Results Structure](#results-structure)
  - [License](#license)

## Overview

This research explores methods for generating consensus statements that fairly represent multiple agent opinions. We formulate text generation as a token-level Markov Decision Process (MDP) and apply social choice theory to balance the opinions of multiple agents. Our framework evaluates different methods for generating fair consensus statements that maximize multi-agent utility across diverse scenarios.

## Methods Implemented

- **Zero Shot**: Baseline that directly prompts for a consensus statement 
- **Best-of-N**: Generates multiple candidate statements and selects the best one
- **Beam Search**: Maintains multiple candidate statements and expands promising ones
- **Finite Lookahead**: Examines a fixed number of future steps to optimize decisions
- **Habermas Machine**: Implements AI-mediated deliberation among participants to find consensus

## Setup and Installation

1. Clone this repository:
```bash
git clone https://github.com/cartgr/Generating-Fair-Consensus-Statements-with-Social-Choice-on-Token-Level-MDPs.git
cd Generating Fair Consensus Statements with Social Choice on Token-Level MDPs
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables by creating a `.env` file:
```
TOGETHER_API_KEY=your_together_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

**Note:** This codebase requires API keys for:
- [Together AI](https://together.ai/) - Used for most language model generation and evaluation
- [OpenAI](https://openai.com/) - Used for LLM-as-judge evaluation and comparative ranking

## Running Experiments

### AAMAS Experiments

The repository includes a dedicated script for running the AAMAS evaluation experiments:

```bash
python run_aamas_experiments.py
```

This script will run all experiments configured in the `configs/appendix/` directory, which contains configurations for:
- Multiple scenarios (1-5)
- Different models (Gemma and Llama)
- Various methods (Habermas, Best-of-N, Beam Search, Finite Lookahead)

You can filter which experiments to run using command-line arguments:

```bash
# Run only Gemma experiments
python run_aamas_experiments.py --model gemma

# Run only experiments for scenario 3
python run_aamas_experiments.py --scenario 3

# Run only Habermas vs. Best-of-N experiments
python run_aamas_experiments.py --method habermas_vs_best_of_n

# Combine filters (e.g., Gemma experiments on scenario 2 with beam search)
python run_aamas_experiments.py --model gemma --scenario 2 --method beam_search
```

Configuration files for the results in the main body can be found in `configs/main_body`

### Custom Experiments

For running custom experiments, use the primary experiment runners:

```bash
# Run experiment with evaluation
python run_experiment_with_eval.py -c configs/your_config.yaml

# Run only the generation phase
python run_experiment.py -c configs/experiment_config.yaml
```

Sample configurations are available in the `configs/examples/` directory:
- `test_best_of_n_config.yaml`: Test Best-of-N decoding
- `test_beam_search_config.yaml`: Test Beam Search decoding
- `test_finite_lookahead_config.yaml`: Test Finite Lookahead decoding
- `test_mcts_config.yaml`: Test MCTS decoding
- `test_habermas_config.yaml`: Test Habermas Machine
- `comprehensive_comparison_config.yaml`: Compare all methods

### Concurrent Execution

The experiment framework supports concurrent execution of methods, which can significantly reduce total experiment runtime:

1. Enable concurrent execution in your config file:
```yaml
# Concurrency Settings
concurrent_execution: true             # Enable/disable concurrent method execution
max_concurrent_methods: 4              # Maximum number of methods to run in parallel
api_rate_limit: 5                      # Maximum API requests per second
```

2. Key concurrency settings:
   - `concurrent_execution`: Toggle parallel execution on/off
   - `max_concurrent_methods`: Control the maximum number of simultaneous methods
   - `api_rate_limit`: Prevent API throttling by limiting request rate

### Running Multiple Seeds

To test the robustness of methods, you can run each experiment with multiple seeds:

1. Set `num_seeds` in your config file:
```yaml
seed: 42
num_seeds: 5  # Will run with seeds 42, 43, 44, 45, 46
```

2. The experiment runner will automatically increment the base seed and run each configuration multiple times.

3. Results from all seeds will be saved in a single CSV file, with the seed value recorded for each run.

## Evaluation Workflow

The experiment workflow is split into two phases:
1. **Generation Phase**: Creating statements using different methods
2. **Evaluation Phase**: Assessing the generated statements

The main benefit of this separation is that it allows us to re-evaluate the results with different models.

### Evaluation Metrics

The evaluation system calculates multiple types of metrics:

1. **Per-agent Utilities**:
   - Average Log Probability (which can be transformed to EPPL): How likely each agent would generate the statement
   - LLM Judge Score (optional): 1-5 score assessing how well the statement represents the opinion

2. **Welfare Metrics**:
   - Egalitarian Welfare: The minimum utility across all agents (max-min fairness)
   - Utilitarian Welfare: The sum of all agents' utilities (maximum total utility)
   - Log Nash Welfare: The sum of log utilities (balances fairness and efficiency)

Each welfare metric is calculated for each utility type (logprob and LLM judge).

### Post-hoc Evaluation

Re-evaluate existing results with a different model:

```bash
python post_hoc_evaluate.py --results-dir results/your_experiment_directory --model new_model_name
```

Example:
```bash
python post_hoc_evaluate.py --results-dir results/deliberation_methods_comparison_20250406_220408 --model google/gemma-2-27b-it
```

Additional options:
```bash
# Skip LLM Judge evaluation (faster, no OpenAI API needed)
python post_hoc_evaluate.py --results-dir results/your_experiment_directory --skip-llm-judge

# Use a specific model for embeddings
python post_hoc_evaluate.py --results-dir results/your_experiment_directory --embedding-model "BAAI/bge-large-en-v1.5"
```

## Results Structure

Experiment results are saved in the `results/` directory, organized by experiment name and timestamp.

The AAMAS experiments specifically output to the `results/appendix/` and `results/main_body` directories. The results for the experiments in the appendix are organized as:
```
results/appendix/
├── AAMAS_gemma_scenario1_beam_search_20250511_222741/
│   ├── config.yaml                # Configuration used
│   ├── results.csv                # Raw results data
│   └── evaluation/                # Evaluation results
│       ├── google_gemma-2-9b-it/  # Results from Gemma evaluator
│       │   ├── seed_0/
│       │   ├── seed_1/
│       │   └── seed_2/
│       ├── improved_aggregate/    # Aggregated metrics
│       └── llm_judge/             # LLM judge evaluations
└── ...
```

Each experiment directory contains:
- `config.yaml`: A copy of the configuration used
- `results.csv`: Results with method parameters, generated statements, and utility scores
- `evaluation/`: Subdirectory with all evaluation results

## License

This project is licensed under the MIT License:

```
MIT License

Copyright (c) 2025 "Generating Fair Consensus Statements with Social Choice on Token-Level MDPs" Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

See [LICENSE](LICENSE) file for the full license text.
