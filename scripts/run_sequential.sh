#!/usr/bin/env bash
# Example sequential run wrapper
if [ "$#" -lt 5 ]; then
  echo "Usage: $0 <n> <c> <m> <k> <method>"
  echo "method: colsample|gaussian"
  exit 1
fi
N=$1
C=$2
M=$3
K=$4
METHOD=$5

python -m src.cli --mode sequential --n $N --c $C --m $M --k $K --method $METHOD
