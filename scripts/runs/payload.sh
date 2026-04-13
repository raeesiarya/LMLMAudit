#!/bin/bash
#SBATCH --job-name=run_audit
#SBATCH --account=fc_cosi
#SBATCH --partition=savio4_gpu
#SBATCH --time=08:00:00
#SBATCH --time-min=06:00:00
#SBATCH --nodes=1

set -euo pipefail

# Sync dependencies from pyproject.toml
uv sync

# Activate uv-managed venv
source .venv/bin/activate

# Run workload
srun python3 src/lmlm-audit/run_audit.py --wandb_activation on