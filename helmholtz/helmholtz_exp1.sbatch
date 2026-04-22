#!/bin/bash
#SBATCH --job-name=helmholtz_exp1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres="gpu:h100-96:1"
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --export=NONE

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi

cd /home/s/simpsone/blend2/helmholtz || exit 1
echo "Working directory: $(pwd)"

export PATH="$HOME/miniforge3/envs/cfd_jax/bin:$PATH"
module load cuda/12.2 cudnn/8.9.5.29

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async

# One-time: install scikit-fem if missing (idempotent).
python -c "import skfem" 2>/dev/null || pip install --no-input scikit-fem

echo "=== Verifying GPU + scikit-fem ==="
python -c "
import tensorflow as tf
print('TF:', tf.__version__, 'GPU:', tf.config.list_physical_devices('GPU'))
import skfem; print('scikit-fem:', skfem.__version__)
"

# ============================================================
# Sweep over wavenumbers k in {4 pi, 8 pi, 12 pi}.
# ============================================================
PI=3.141592653589793
for MULT in 4 8 12; do
    K=$(python -c "print(${MULT} * ${PI})")
    TAG="k${MULT}pi"
    echo ""
    echo "=============================================================="
    echo "=== Experiment 1: k = ${MULT} pi  (k=${K})  tag=${TAG}"
    echo "=============================================================="

    python train_pinn.py \
        --k ${K} \
        --epochs 20000 \
        --lr 3e-4 \
        --n-domain 10000 \
        --n-boundary 2000 \
        --w-bc 100.0 \
        --tag ${TAG}

    python run_fem.py \
        --k ${K} \
        --mesh-n 129 \
        --nx 201 --ny 201 \
        --tag ${TAG}

    python plot_sanity.py \
        --k ${K} \
        --tag ${TAG}
done

echo "=== Job completed ==="
