#!/bin/bash
#SBATCH --job-name=Nystrom_Seq_5_Srht
#SBATCH --account=p200981
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/seq_%j.out

set -e
mkdir -p logs

if [ -f /etc/profile ]; then source /etc/profile; fi
if ! command -v module >/dev/null 2>&1; then [ -f /etc/profile.d/lmod.sh ] && source /etc/profile.d/lmod.sh; fi
if ! command -v module >/dev/null 2>&1; then [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh; fi

module --force purge
module load env/release/2024.1
module load Python/3.12.3-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a

set -euo pipefail

source venv/bin/activate

if [ -z "${EBROOTPYTHON:-}" ] || [ ! -d "${EBROOTPYTHON}/lib" ]; then
  echo "ERROR: EBROOTPYTHON not set correctly."
  exit 1
fi
export LD_LIBRARY_PATH="${EBROOTPYTHON}/lib:${LD_LIBRARY_PATH:-}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUT_DIR="results/sequential_${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

N=5000
C=100000
L_LIST="600,1000,2000,2500,3000"
K_LIST="100,200,300,400,500,600,700,800,900,1000"

python -m src.cli \
  --dataset msd \
  --n "${N}" \
  --c "${C}" \
  --l-list "${L_LIST}" \
  --k-list "${K_LIST}" \
  --sketch srht \
  --out-dir "${OUT_DIR}"
