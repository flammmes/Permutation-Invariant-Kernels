# Permutation-Invariant Bayesian Optimization for CCS Well Design

This repository contains research code for Bayesian Optimization (BO) with permutation-invariant surrogate models for Carbon Capture and Storage (CCS) well-design problems.

The core setting is BO over structured designs containing:

- auxiliary continuous controls, e.g. group injection and production targets,
- an unordered set of injector well locations,
- an unordered set of producer well locations.

Under group control, the objective is invariant to permutations within the injector group and within the producer group. This repository compares standard GP/DKL surrogates with permutation-aware alternatives.

## Main models

- **GP-Perm**: a Gaussian Process surrogate with a permutation-invariant kernel based on set divergences between injector, producer, and injector-producer interaction sets.
- **DKL-DS**: a Deep Kernel Learning baseline that uses Deep Sets encoders to learn permutation-invariant embeddings.
- **Set-kernel baselines**: double-sum and deep-embedding style set kernels.
- **Non-invariant baselines**: standard GP and DKL models on flattened vector inputs.

## Repository layout

```text
.
├── main_v3.py
├── models/
│   ├── __init__.py
│   ├── gp.py
│   ├── gp_perm_inv_var4.py
│   ├── dklsets_v4.py
│   └── gp_set_kernels.py
├── test_functions/
│   ├── __init__.py
│   ├── johansen.py
│   └── synthetic_test_functions.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Create a fresh Python environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU runs, install the PyTorch build that matches your CUDA version before installing the remaining dependencies.

## Quick start

Run the main experiment script:

```bash
python main_v3.py
```

If `main_v3.py` accepts command-line arguments, document the exact commands here, for example:

```bash
# Example only; update according to the actual argparse options in main_v3.py
python main_v3.py --model gp_perm --test_function synthetic --seed 0
```

## Synthetic benchmarks

Synthetic objectives are implemented in:

```text
test_functions/synthetic_test_functions.py
```

These are useful for quick checks because they do not require the reservoir simulator.

## Johansen CCS case study

The Johansen objective is implemented in:

```text
test_functions/johansen.py
```

The full simulator-based workflow may require external assets and software, such as reservoir model files and OPM Flow. Large simulator inputs/outputs are intentionally not tracked in git. Use a local `data/`, `runs/`, or `simulation_outputs/` directory, or use Git LFS for any files that must be versioned.

## Development checks before pushing

Run a lightweight syntax check:

```bash
python -m compileall main_v3.py models test_functions
```

Check for hard-coded local paths or secrets before committing:

```bash
grep -R "C:\\\|/home/\|/mnt/\|.pem\|AWS\|SECRET\|TOKEN" -n . \
  --exclude-dir=.git --exclude-dir=.venv
```

## Citation

Citation information will be added after the paper is public.

```bibtex
@misc{permutation_invariant_bo_ccs,
  title  = {Inducing Permutation Invariant Priors in Bayesian Optimization for Carbon Capture and Storage Applications},
  author = {Anonymous},
  year   = {2025},
  note   = {Under review}
}
```

## License

Add a license before making the repository public. Common choices for research code are MIT, BSD-3-Clause, or Apache-2.0, but confirm with all co-authors and your institution.
