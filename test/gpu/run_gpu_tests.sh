#!/bin/bash -l
# The GPU test suite on one A100 of albedo. Instantiate the two environments on a login node first (the compute nodes
# are offline): julia --project=test/gpu/env_threads -e "using Pkg; Pkg.instantiate()", and the same for env_cuda.
# Submit from the repository root: sbatch test/gpu/run_gpu_tests.sh
#SBATCH --job-name=shakti_gpu_test
#SBATCH --account=envi.envi
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --qos=30min
#SBATCH --time=00:30:00
#SBATCH --chdir=/albedo/home/taange001/Shakti.jl
#SBATCH --output=test/gpu/logs/%x_%j.out
#SBATCH --error=test/gpu/logs/%x_%j.out

set -euo pipefail
nvidia-smi -L
JULIA_PKG_OFFLINE=true julia -t 4 test/gpu/runtests.jl
