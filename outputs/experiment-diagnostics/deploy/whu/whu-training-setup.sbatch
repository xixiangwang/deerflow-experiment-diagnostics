#!/bin/bash
# CPU-only offline dependency installation; submit with your CPU account and partition.
#SBATCH --job-name=deerflow-torch-setup
#SBATCH --partition=hpxg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=training-setup-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
test -n "${SLURM_JOB_ID:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PIP_CACHE_DIR="$PWD/state/pip-cache"
export HF_HOME="$PWD/state/huggingface"
export PIP_DISABLE_PIP_VERSION_CHECK=1
BASE="$PWD/state/python312-7d6ef1164fb9/python/bin/python3.12"
"$BASE" install_training.py
APP="$("$BASE" -c 'import json; print(json.load(open("state/installation.json"))["application"])')"
FRAMEWORK="$APP/work/runtime-py312/bin/python"
TRAINING="$PWD/state/training-cu121-py312/bin/python"
"$FRAMEWORK" "$APP/outputs/experiment-diagnostics/launch.py" self-test --torch-python "$TRAINING"
"$FRAMEWORK" "$APP/outputs/experiment-diagnostics/launch.py" pretrained-check --torch-python "$TRAINING" --local-model "$PWD/models/Qwen2.5-0.5B-Instruct" > state/training-preflight.json
