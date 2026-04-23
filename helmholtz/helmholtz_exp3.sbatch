#!/bin/bash
#SBATCH --job-name=helmholtz_exp3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres="gpu:h100-96:1"
#SBATCH --mem=64G
#SBATCH --time=24:00:00
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
# Exp 3: parametric PINN u(x, y, k, x_s, y_s) + multi-config
# router + hybrid eval across ID/OOD configs.
# ============================================================
PINN_TAG="exp3_parametric"
ROUTER_TAG="exp3"
CONFIGS="configs_exp3.json"
PREFIX="exp3"

PI=3.141592653589793
K_MIN=$(python -c "print(2.0 * ${PI})")
K_MAX=$(python -c "print(6.0 * ${PI})")

echo ""
echo "=============================================================="
echo "=== Exp 3  parametric   k in [${K_MIN}, ${K_MAX}]"
echo "=============================================================="

# 1) Parametric PINN
python train_pinn_parametric.py \
    --k-min ${K_MIN} --k-max ${K_MAX} \
    --xs-min 0.05 --xs-max 0.95 \
    --ys-min 0.05 --ys-max 0.95 \
    --sigma 0.05 --amplitude 250.0 \
    --hole-center-x 0.5 --hole-center-y 0.5 --hole-radius 0.15 \
    --epochs 80000 \
    --lr 3e-4 \
    --n-configs 16 \
    --n-domain-per-config 1024 \
    --n-boundary-per-config 256 \
    --w-bc 100.0 \
    --layers 256 256 256 256 256 \
    --fourier-m 128 \
    --resample-every 100 \
    --tag ${PINN_TAG}

# 2) FEM references for every config
python run_fem_multi.py \
    --configs ${CONFIGS} \
    --tag-prefix ${PREFIX} \
    --parametric-meta ./history/training_meta_${PINN_TAG}.json \
    --mesh-n 129 --nx 201 --ny 201

# 3) Train shared router on ID configs
python train_router_multi.py \
    --pinn-tag ${PINN_TAG} \
    --configs ${CONFIGS} \
    --tag ${ROUTER_TAG} \
    --epochs 4000 \
    --lr 1e-3 \
    --beta 0.1 \
    --lambda-tv 0.01 \
    --base-filters 32

# 4) Multi-config timing + coverage sweep (ID + OOD)
python time_helmholtz_multi.py \
    --pinn-tag ${PINN_TAG} \
    --router-tag ${ROUTER_TAG} \
    --configs ${CONFIGS} \
    --tag-prefix ${PREFIX} \
    --mesh-n 129 --nx 201 --ny 201

# 5) Per-config plots + summary table
python plot_hybrid_multi.py \
    --pinn-tag ${PINN_TAG} \
    --router-tag ${ROUTER_TAG} \
    --configs ${CONFIGS} \
    --tag-prefix ${PREFIX}

echo "=== Job completed ==="
