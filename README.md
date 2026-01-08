# Randomized Nyström Approximation 

This repository contains a scaffold for experiments with randomized Nyström low-rank approximations, dataset construction from MNIST using mnist_datasets package, and a sequential + MPI implementation outline.

Structure
- `src/` : core modules (`dataset.py`, `nystrom.py`, `mpi_nystrom.py`, `cli.py`)
- `scripts/` : helper scripts to run experiments
- `requirements.txt` : Python dependencies

Quick start

1. Create a Python environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Example sequential run (build A from MNIST, compute Nyström):

```bash
python -m src.cli --mode sequential --n 2000 --c 100 --m 500 --k 50
```

3. Example MPI run (requires `mpirun` and `mpi4py`):

```bash
mpirun -n 4 python -m src.cli --mode mpi --n 4000 --c 1e5 --m 800 --k 50
```

4. Example sweep run (vary m and k, save results and plots):

```bash
python -m src.cli --sweep --n 1000 --c 100 --m-list 100,200,400 --k-list 50,100,200 --method gaussian --out-dir results/experiment --debug 
```

This builds the RBF kernel once, computes Nyström approximations for each m, evaluates relative nuclear errors for each k, and saves `results.csv`, `meta.json`, and plots (`rel_error_vs_k.png`, `nuc_vs_k.png`) in a timestamped subdirectory under `results/experiment_YYYYMMDD_HHMMSS`. Use `--debug` for detailed timing prints, `--approx-nuclear` for faster (approximate) nuclear norm computation.

Notes
- Dataset: MNIST (training or test set via --train/--no-train)
- Kernel: radial basis function $\exp(-||xi-xj||^2 / c^2)$.
- Default sketch: column sampling (random subset) and Gaussian sketch option available.
- For large `n`, constructing the dense `n x n` kernel may be memory intensive—choose `n` accordingly.

See `src/` for implementation details and options.
