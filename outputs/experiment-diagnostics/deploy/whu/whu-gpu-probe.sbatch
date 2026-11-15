#!/bin/bash
# Submit explicitly with a confirmed billing account. This allocates ONE GPU for diagnostics only.
#SBATCH --job-name=deerflow-gpu-probe
#SBATCH --partition=a100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=gpu-probe-%j.log

set -eu
test -n "${SLURM_JOB_ID:-}"
printf 'Node: %s\n' "$(hostname)"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
printf 'SLURM_JOB_GPUS: %s\n' "${SLURM_JOB_GPUS:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
