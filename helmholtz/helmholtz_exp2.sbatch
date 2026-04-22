#!/bin/bash
#SBATCH --job-name=helmholtz_exp2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres="gpu:h100-96:1"
#SBATCH --mem=32G
#SBATCH --time=10:00:00
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

python -c "import skfem" 2>/dev/null || pip install --no-input scikit-fem

echo "=== Verifying GPU + scikit-fem ==="
python -c "
import tensorflow as tf
print('TF:', tf.__version__, 'GPU:', tf.config.list_physical_devices('GPU'))
import skfem; print('scikit-fem:', skfem.__version__)
"

# ============================================================
# Exp 2: square-with-hole, Gaussian source at (0.7, 0.5), k=4pi.
# Train PINN -> FEM reference -> Router -> Timing+sweep -> Plots.
# ============================================================
PI=3.141592653589793
K=$(python -c "print(4.0 * ${PI})")
TAG="exp2_hole_k4pi"

XS=0.7
YS=0.5
SIGMA=0.05
AMP=250.0
HCX=0.5
HCY=0.5
HR=0.15

echo ""
echo "=============================================================="
echo "=== Exp 2  k=4pi (${K})  tag=${TAG}"
echo "=============================================================="

# 1) PINN
python train_pinn.py \
    --k ${K} \
    --domain square_hole \
    --hole-center-x ${HCX} --hole-center-y ${HCY} --hole-radius ${HR} \
    --source gaussian \
    --x-s ${XS} --y-s ${YS} --sigma ${SIGMA} --amplitude ${AMP} \
    --epochs 50000 \
    --lr 3e-4 \
    --n-domain 20000 \
    --n-boundary 4000 \
    --w-bc 100.0 \
    --layers 256 256 256 256 256 \
    --fourier-m 128 \
    --tag ${TAG}

# 2) FEM reference
python run_fem.py \
    --k ${K} \
    --domain square_hole \
    --hole-center-x ${HCX} --hole-center-y ${HCY} --hole-radius ${HR} \
    --source gaussian \
    --x-s ${XS} --y-s ${YS} --sigma ${SIGMA} --amplitude ${AMP} \
    --mesh-n 129 \
    --nx 201 --ny 201 \
    --tag ${TAG}

# 3) Router
python train_router.py \
    --k ${K} \
    --domain square_hole \
    --hole-center-x ${HCX} --hole-center-y ${HCY} --hole-radius ${HR} \
    --source gaussian \
    --x-s ${XS} --y-s ${YS} --sigma ${SIGMA} --amplitude ${AMP} \
    --epochs 2000 \
    --lr 1e-3 \
    --beta 1.1 \
    --lambda-tv 0.01 \
    --base-filters 32 \
    --tag ${TAG}

# 4) Timing + coverage sweep
python time_helmholtz.py \
    --k ${K} \
    --domain square_hole \
    --hole-center-x ${HCX} --hole-center-y ${HCY} --hole-radius ${HR} \
    --source gaussian \
    --x-s ${XS} --y-s ${YS} --sigma ${SIGMA} --amplitude ${AMP} \
    --mesh-n 129 \
    --nx 201 --ny 201 \
    --tag ${TAG}

# 5) Plots + metrics
python plot_hybrid.py --tag ${TAG}

echo "=== Job completed ==="
