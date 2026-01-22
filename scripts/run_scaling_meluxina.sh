#!/bin/bash
#SBATCH --job-name=Nystrom_10kScaling
#SBATCH --account=p200981
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/10k_%j.out

set -e
mkdir -p logs

# --- Initialize module system on compute nodes ---
if [ -f /etc/profile ]; then
  source /etc/profile
fi
if ! command -v module >/dev/null 2>&1; then
  [ -f /etc/profile.d/lmod.sh ] && source /etc/profile.d/lmod.sh
fi
if ! command -v module >/dev/null 2>&1; then
  [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
fi

set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
  echo "ERROR: 'module' command not found on this node."
  exit 1
fi

echo "=== JOB INFO ==="
echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID:-}"
echo "Requested ntasks: ${SLURM_NTASKS:-}"
echo "==============="

module --force purge
module load env/release/2024.1
module load Python/3.12.3-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a
module load OpenMPI/5.0.3-GCC-13.3.0

echo "=== MODULE PYTHON ==="
which python
python -V
python -c "import sys; print(sys.executable)"
echo "====================="

if [ ! -f venv/bin/activate ]; then
  echo "ERROR: venv/bin/activate not found."
  exit 1
fi
source venv/bin/activate

echo "=== VENV PYTHON ==="
which python
python -V
python -c "import sys; print(sys.executable)"
echo "==================="

if [ -z "${EBROOTPYTHON:-}" ] || [ ! -d "${EBROOTPYTHON}/lib" ]; then
  echo "ERROR: EBROOTPYTHON is not set correctly (modules not loaded?)."
  echo "EBROOTPYTHON='${EBROOTPYTHON:-}'"
  exit 1
fi
export LD_LIBRARY_PATH="${EBROOTPYTHON}/lib:${LD_LIBRARY_PATH:-}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUT_DIR="results/scaling_N40k_${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

SUMMARY_CSV="${OUT_DIR}/scaling_summary.csv"
echo "Summary CSV: ${SUMMARY_CSV}"

echo "=== HEAVY SCALING (N=40000) ==="
echo "Output: ${OUT_DIR}"
echo "==============================="

FIXED_N=40000
FIXED_C=100000
FIXED_L=2000
FIXED_K=500

PROCS_LIST=(1 4 9 16 36 64)

for P in "${PROCS_LIST[@]}"; do
  CUR_DIR="${OUT_DIR}/p${P}"
  mkdir -p "${CUR_DIR}"

  GRID=$(awk -v p="$P" 'BEGIN{g=int(sqrt(p)); if (g*g==p) printf("%dx%d", g, g); else printf("na");}')

  echo "------------------------------------------------"
  echo "Running with ${P} MPI tasks (grid ${GRID})"
  echo "Output dir: ${CUR_DIR}"

  srun --ntasks="${P}" --kill-on-bad-exit=1 --export=ALL,LD_LIBRARY_PATH \
    python -m src.cli \
      --dataset msd \
      --n "${FIXED_N}" \
      --c "${FIXED_C}" \
      --l-list "${FIXED_L}" \
      --k-list "${FIXED_K}" \
      --sketch gaussian \
      --mpi \
      --out-dir "${CUR_DIR}" \
      --skip-error \
      --summary-csv "${SUMMARY_CSV}"

      

  echo "Done: P=${P}"
done

echo "=== DONE ==="
echo "CSV written to: ${SUMMARY_CSV}"
