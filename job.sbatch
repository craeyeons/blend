#!/bin/bash
#SBATCH --gres="gpu:a100-80:1"
#SBATCH --job-name=cfd_jax
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-long
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Setup conda if not already done
if [ ! -d "$HOME/miniforge3" ]; then
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh -b
fi

export PATH="$HOME/miniforge3/bin:$PATH"
source $HOME/miniforge3/etc/profile.d/conda.sh

# Create environment if it doesn't exist
if ! conda env list | grep -q "cfd_jax"; then
    conda create -y -n cfd_jax python=3.10
fi

conda activate cfd_jax

# Install JAX with CUDA support (simpler than FEniCS+PETSc GPU setup)
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install matplotlib numpy

# Verify GPU is available
python -c "import jax; print('JAX devices:', jax.devices())"

echo "Running JAX CFD simulation..."
python main.py
