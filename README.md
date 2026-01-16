# Randomized Nyström Approximation 

This project implements randomized Nyström low-rank approximations for kernel matrices with both sequential and distributed (MPI) support. It includes dataset loading (MNIST and Year Prediction MSD), RBF kernel computation, and comprehensive sweep experiments.

## Structure

- `src/` : Core modules
  - `cli.py` : Command-line interface and sweep orchestration
  - `dataset.py` : Data loading (MNIST, MSD) and RBF kernel computation
  - `nystrom.py` : Sequential Nyström implementation with sketching methods
  - `mpi_nystrom.py` : MPI-based distributed Nyström implementation
  - `error.py` : Error computation (relative nuclear norm)
  - `analysis.py` : Results visualization and saving
  - `sketch.py` : Sketching matrix generation (Gaussian, SRHT)
- `scripts/` : Helper shell scripts
- `results/` : Output directory for experiments (auto-created with timestamps)
- `datasets/` : Data directory (auto-created if needed)

## Installation

```bash
# Create and activate Python environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For MPI support, also install mpi4py
pip install mpi4py
```

## Usage

The CLI runs **sweep experiments only** - it varies sketch size `m` and rank `k` over specified ranges and saves results.

### Basic Command Structure

```bash
python -m src.cli \
  --n <num_samples> \
  --c <bandwidth> \
  --l-list <l1,l2,...> \
  --k-list <k1,k2,...> \
  [OPTIONS]
```

### Required Arguments

| Flag | Type | Description |
|------|------|-------------|
| `--n` | int | Number of samples to load from dataset |
| `--c` | float | RBF bandwidth parameter (c in exp(-\|\|x_i - x_j\|\|² / c²)) |
| `--l-list` | str | Comma-separated sketch sizes (e.g., `100,200,400`) |
| `--k-list` | str | Comma-separated target ranks (e.g., `50,100,200`) |

### Optional Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dataset` | str | `mnist` | Dataset to use: `mnist` or `msd` |
| `--dataset-path` | str | `./datasets/YearPredictionMSD.txt` | Path to MSD dataset file |
| `--train` | flag | False | Use MNIST training set (default: train set) |
| `--sketch` | str | `gaussian` | Sketching method: `gaussian` or `srht` |
| `--seed` | int | `0` | Random seed for reproducibility |
| `--out-dir` | str | `results` | Output directory for results |
| `--debug` | flag | False | Enable debug prints with detailed timing |
| `--mpi` | flag | False | Run with MPI for distributed computation |

## Examples

### Sequential Sweep (Single Machine)

Basic sequential run with Gaussian sketching:
```bash
python -m src.cli \
  --n 1000 \
  --c 100000 \
  --l-list 100,200,400 \
  --k-list 50,100,200 \
  --sketch gaussian \
  --dataset msd
```

Using MNIST dataset with debug output:
```bash
python -m src.cli \
  --n 5000 \
  --c 100 \
  --l-list 200,400 \
  --k-list 100,200 \
  --sketch gaussian \
  --dataset mnist \
  --train \
  --debug
```

### Parallel Sweep (MPI)

Run with 4 processors (must be a perfect square: 1, 4, 9, 16, 25, etc.):
```bash
mpirun -np 4 python -m src.cli \
  --n 4000 \
  --c 100000 \
  --l-list 100,200,400 \
  --k-list 50,100,200 \
  --sketch gaussian \
  --dataset msd \
  --mpi
```

With 9 processors and SRHT sketching:
```bash
mpirun -np 9 python -m src.cli \
  --n 10000 \
  --c 100000 \
  --l-list 200,400,800 \
  --k-list 100,200 \
  --sketch srht \
  --dataset msd \
  --mpi \
  --debug
```

## Output

Results are saved to `results/YYYYMMDD_HHMMSS/` with:

- **results.csv** : Detailed results table with columns:
  - `n`, `c`, `l`, `k` : Experiment parameters
  - `dataset` : Dataset used
  - `sketch_method` : Sketching method (gaussian or srht)
  - `compute_time` : Time to compute U_hat and Λ_k (seconds)
  - `rel_error` : Relative nuclear norm error

- **meta.json** : Metadata about the experiment
- **rel_error_vs_k.png** : Plot of relative error vs rank k (curves for each m)

### Example Output

```
Loading MSD set (n=1000)
[DEBUG] Load time: 0.0532s
Building RBF kernel matrix A
[DEBUG] Kernel build time: 0.1234s, shape: (1000, 1000), size: 0.01 GB
Nuclear norm of A: 1.000000e+03

--- Running m=100 ---
m=100, k=50: Time=0.0239s Error=2.041948e-05
m=100, k=100: Time=0.0239s Error=6.309718e-06

--- Running m=200 ---
m=200, k=50: Time=0.0539s Error=1.750360e-05
m=200, k=100: Time=0.0539s Error=1.471730e-06

Saved results and plots to results/20260116_162659
```

## Sketching Methods

- **Gaussian** (`--sketch gaussian`): Generates dense Gaussian random matrix Ω with entries ~ N(0,1). Fast and simple.
- **SRHT** (`--sketch srht`): Subsampled Randomized Hadamard Transform. More structured, potentially faster for very large matrices.

## Datasets

- **MNIST** : 60,000 training + 10,000 test images. Use `--train` for training set, omit for test set.
- **MSD** : Year Prediction MSD dataset with 90 audio features. Auto-downloads from UCI if not found.

## Notes

- For MPI runs, the number of processors must be a perfect square (1, 4, 9, 16, 25, etc.) due to 2D block distribution.
- The kernel matrix A is n × n, so for large n, memory usage is O(n²). Choose n accordingly for your system.
- Use `--debug` for detailed timing information on individual components.
- Results are automatically timestamped and saved, allowing multiple experiments without overwriting.

## Implementation Details

See `src/` for:
- **nystrom.py** : Sequential Nyström with configurable sketching
- **mpi_nystrom.py** : Distributed version using 2D block grid of processors
- **sketch.py** : Sketching matrix generation functions

