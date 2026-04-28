#!/bin/bash
#SBATCH --job-name=helmholtz_exp3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres="gpu:h100-96:1"
#SBATCH --mem=64G
#SBATCH --time=24:00:00

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

python -c "import skfem" 2>/dev/null || pip install --no-input scikit-fem
python -c "
import tensorflow as tf
print('TF:', tf.__version__, 'GPU:', tf.config.list_physical_devices('GPU'))
import skfem; print('scikit-fem:', skfem.__version__)
"

# ============================================================
# Exp 3: 2D Poisson with circular hole. N independent single-case
# PINNs (one per source position) + ONE shared router trained
# across all of them, evaluated on held-out OOD configs (each
# with its own pre-trained PINN). Same code path as Helmholtz
# but with k=0; the directory name is historical.
# ============================================================
CONFIGS="configs_exp3.json"
TAG="exp3"

echo ""
echo "=============================================================="
echo "=== Exp 3 (multi-PINN + shared router)  tag=${TAG}"
echo "=============================================================="

# Stage selector — set this when submitting (sbatch --export=STAGE=analyze ...).
#   all      — PINN training (1) + router training (2a) + analysis (2b)
#   train    — PINN training (1) + router training (2a) only
#   router   — router training (2a) only (PINNs assumed present)
#   analyze  — analysis (2b) only (router + PINNs assumed present)
STAGE=${STAGE:-all}
echo "STAGE=${STAGE}"

# 1) Train one scalar PINN per config (train split AND test split —
#    test PINNs are needed so the router sees a novel PINN at eval).
if [[ "${STAGE}" == "all" || "${STAGE}" == "train" ]]; then
    python train_pinns_multi.py \
        --configs ${CONFIGS} \
        --also-train-test \
        --skip-existing \
        --epochs 15000 \
        --lr 3e-4 \
        --n-domain 20000 \
        --n-boundary 4000 \
        --w-bc 100.0 \
        --layers 256 256 256 256 256 \
        --fourier-m 64 \
        --sigma 0.05 --amplitude 250.0 \
        --hole-center-x 0.5 --hole-center-y 0.5 --hole-radius 0.15 \
        --grad-clip 1.0
fi

# 2a) Train shared router (no analysis).
if [[ "${STAGE}" == "all" || "${STAGE}" == "train" || "${STAGE}" == "router" ]]; then
    python train_router_multi.py \
        --configs ${CONFIGS} \
        --tag ${TAG} \
        --epochs 2000 \
        --lr 1e-3 \
        --beta 1.0 \
        --lambda-tv 1.0 \
        --base-filters 32 \
        --mesh-n 129 --nx 201 --ny 201 \
        --sigma 0.05 --amplitude 250.0 \
        --hole-center-x 0.5 --hole-center-y 0.5 --hole-radius 0.15 \
        --skip-analysis
fi

# 2b) Run full analysis using the saved router (FEM refs, coverage sweep,
#     per-config plots, summary CSV + MD).
if [[ "${STAGE}" == "all" || "${STAGE}" == "analyze" ]]; then
    python train_router_multi.py \
        --configs ${CONFIGS} \
        --tag ${TAG} \
        --base-filters 32 \
        --mesh-n 129 --nx 201 --ny 201 \
        --sigma 0.05 --amplitude 250.0 \
        --hole-center-x 0.5 --hole-center-y 0.5 --hole-radius 0.15 \
        --skip-train
fi

echo "=== Job completed ==="
