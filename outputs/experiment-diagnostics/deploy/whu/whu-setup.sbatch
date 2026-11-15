#!/bin/bash
# Submit from the unpacked whu-deerflow directory with: sbatch -A CONFIRMED_ACCOUNT whu-setup.sbatch
#SBATCH --job-name=deerflow-setup
#SBATCH --partition=hpib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=setup-%j.log

set -eu
cd "${SLURM_SUBMIT_DIR:?Submit through Slurm}"
test -f bundle-manifest.json
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
python3 install_whu.py --install
