#!/usr/bin/env bash
# Example MPI run wrapper (run with mpirun/mpiexec)
if [ "$#" -lt 6 ]; then
  echo "Usage: $0 <nprocs> <n> <c> <m> <k> <method>"
  echo "Example: $0 4 4000 1e5 800 50 colsample"
  exit 1
fi
NPROCS=$1
N=$2
C=$3
M=$4
K=$5
METHOD=$6

mpirun -n $NPROCS python -m src.cli --mode mpi --n $N --c $C --m $M --k $K --method $METHOD
