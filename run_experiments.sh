#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$(pwd)}"
if [[ $# -ge 1 ]]; then
  shift || true
fi
EXTRA_ARGS=("$@")

TRAIN_SCRIPT="$PROJECT_DIR/scripts/train_mujoco.py"
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "找不到训练脚本: $TRAIN_SCRIPT" >&2
  echo "请将本脚本放在包含 scripts/train_mujoco.py 的工程根目录，或传入工程路径作为参数。" >&2
  exit 1
fi

# ===== 配置区 =====
# 3 个种子
SEEDS=(100 200 300)

# 要使用的 GPU（默认 3 张卡：0,1,2）
IFS=',' read -r -a GPUS <<< "${GPUS:-0,1,2}"
MAX_JOBS="${#GPUS[@]}"

# 显存占用比例（可通过环境变量 MEM_FRAC 覆盖）
MEM_FRAC="${MEM_FRAC:-0.90}"

# conda 环境名
CONDA_ENV="${CONDA_ENV:-relax}"

# 日志目录
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"
# ==================

run_one() {
  local seed="$1"
  local gpu="$2"
  local log_file="${LOG_DIR}/seed_${seed}.log"

  echo "[`date '+%F %T'`] 启动 seed=${seed} -> GPU ${gpu} | 日志: ${log_file}"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  XLA_FLAGS='--xla_gpu_deterministic_ops=true' \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRAC}" \
  conda run -n "${CONDA_ENV}" --no-capture-output \
    python "$TRAIN_SCRIPT" --seed "$seed" "${EXTRA_ARGS[@]}" \
    >"${log_file}" 2>&1
}

i=0
for seed in "${SEEDS[@]}"; do
  gpu="${GPUS[$(( i % MAX_JOBS ))]}"

  # 启动后台任务
  run_one "$seed" "$gpu" &

  # 限制并发：最多同时跑 MAX_JOBS 个
  while [[ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]]; do
    sleep 2
  done

  ((i+=1))
done

# 等待所有任务完成
wait
echo "== 所有种子完成 =="
