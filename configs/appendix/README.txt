# NeurIPS Experiments Setup

This directory contains configurations for the NeurIPS paper experiments, structured to compare different deliberation methods across 5 different scenarios using both Gemma and Llama models.

## Directory Structure

```
aamas/
├── README.txt (this file)
├── gemma/ (Experiments using Gemma-2-9b-it as generation model)
│   ├── scenario_1/ (Genetic code privacy)
│   │   ├── habermas_vs_best_of_n.yaml
│   │   ├── habermas_only.yaml
│   │   ├── beam_search.yaml
│   │   └── finite_lookahead.yaml
│   ├── scenario_2/ (Tax and benefits system)
│   │   └── ... (same as scenario_1)
│   ├── scenario_3/ (Alcohol and cigarettes ban)
│   │   └── ... (same as scenario_1)
│   ├── scenario_4/ (Children's education views)
│   │   └── ... (same as scenario_1)
│   └── scenario_5/ (UK and EU)
│       └── ... (same as scenario_1)
└── llama/ (Experiments using Llama-3.1-8B-Instruct-Turbo as generation model)
    ├── scenario_1/
    │   └── ... (same as gemma)
    ├── scenario_2/
    │   └── ... (same as gemma)
    ├── scenario_3/
    │   └── ... (same as gemma)
    ├── scenario_4/
    │   └── ... (same as gemma)
    └── scenario_5/
        └── ... (same as gemma)
```

## Experiment Configurations

Each scenario contains 4 experiment types:

1. **Habermas vs Best-of-N** (`habermas_vs_best_of_n.yaml`):
   - Compares Habermas Machine (1 round) with Best-of-N
   - N parameter / num_candidates: [1, 3, 5, 10, 20, 50]

2. **Habermas Machine Only** (`habermas_only.yaml`):
   - Num rounds: [1, 2]
   - Num candidates: [2, 5, 10]

3. **Beam Search** (`beam_search.yaml`):
   - Beam width: [2, 4, 6, 8, 10]
   - Max tokens: 100
   - Brush up model: Same as generation model

4. **Finite Lookahead** (`finite_lookahead.yaml`):
   - Branching factor: [2, 3, 4]
   - Depth: [1, 2, 3]
   - Max tokens: 100
   - Brush up model: Same as generation model

## Common Settings

- All experiments are run with 3 different random seeds
- Each model variant (gemma and llama) uses different generation models but both use Gemma and Llama for evaluation
- Concurrency: 10 threads with API rate limit of 10

## Scenarios

1. Should a person's genetic code be considered private information?
2. Should we increase taxes to fund a more comprehensive benefits system?
3. Should we ban the sale of alcohol and cigarettes in public places?
4. Are children's views about their education important?
5. Is the UK better off inside or outside of the European Union?

## Running Experiments

To run all NeurIPS experiments, use the `run_aamas_experiments.py` script:

```
python run_aamas_experiments.py
```

This script will:
1. Create all necessary output directories in `results/aamas/`
2. Run all experiments sequentially across both models and all scenarios
3. Perform evaluations using both evaluation models
4. Generate aggregate metrics for analysis

You can run specific experiment groups with optional flags:

```
# Run only Gemma experiments
python run_aamas_experiments.py --model gemma

# Run only a specific scenario 
python run_aamas_experiments.py --scenario 1

# Run only a specific method type
python run_aamas_experiments.py --method beam_search
```

All results will be stored in a corresponding structure under `results/aamas/`.

## Analysis

After running all experiments, consolidated results will be available in:
- `results/aamas/gemma/analysis/`
- `results/aamas/llama/analysis/`

These directories will contain aggregated metrics and comparison charts.