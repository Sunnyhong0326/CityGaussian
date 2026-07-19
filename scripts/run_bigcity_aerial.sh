#!/usr/bin/env bash
#
# Full BigCity aerial pipeline (stages 1-6), mirroring scripts/run_citygs.sh:
#   1. coarse global Gaussian model      (config/noncolmap/bigcity_aerial_coarse.yaml)
#   2. data partitioning                 (config/noncolmap/bigcity_aerial_c36.yaml)
#   3. per-block fine-tuning             (block_dim [11,11,1] -> 121 blocks, ids 0..120)
#   4. merge blocks into a single model
#   5. render the test split             (render_large.py --eval --skip_train)
#   6. metrics                           (metrics_large.py -t test --use_cc)
#
# Test split handling:
#   render_large.py now has an --eval flag. With --eval, the MatrixCity reader
#   loads transforms_test.json from the config's source_path (pose/all_blocks)
#   as the test cameras; --skip_train additionally skips parsing the ~51k
#   train frames at startup, so only the 9k test frames are read. Renders go
#   to <model_path>/test/ours_<iter>/, which stage 6 evaluates with -t test.
#
# Usage:
#   bash scripts/run_bigcity_aerial.sh
#
# Env-var overrides (all optional):
#   COARSE_CONFIG   default: noncolmap/bigcity_aerial_coarse
#   CONFIG          default: noncolmap/bigcity_aerial_c36
#   MAX_BLOCK_ID    default: derived from block_dim in $CONFIG (121 blocks -> 120)
#   MEM_THRESHOLD   default: 5000 (MiB used before a GPU counts as "free")
#   SLEEP_BETWEEN   default: 60   (seconds between block launches)
#   START_PORT      default: 4041
#   SKIP_COARSE     "1" to skip stage 1 (resume from an existing coarse ckpt)
#   SKIP_PARTITION  "1" to skip stage 2
#   SKIP_BLOCKS     "1" to skip stage 3
#   SKIP_MERGE      "1" to skip stage 4
#   SKIP_RENDER     "1" to skip stage 5
#   SKIP_METRICS    "1" to skip stage 6

set -euo pipefail

COARSE_CONFIG="${COARSE_CONFIG:-noncolmap/bigcity_aerial_coarse}"
CONFIG="${CONFIG:-noncolmap/bigcity_aerial_c121}"
MEM_THRESHOLD="${MEM_THRESHOLD:-5000}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-60}"
START_PORT="${START_PORT:-4041}"

out_name="test"   # render_large.py --eval writes renders under <model_path>/test/

# Derive MAX_BLOCK_ID from block_dim in the config unless overridden.
if [[ -z "${MAX_BLOCK_ID:-}" ]]; then
  MAX_BLOCK_ID=$(python3 - "config/${CONFIG}.yaml" <<'EOF'
import sys, yaml
cfg = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)
d = cfg["model_params"]["block_dim"]
print(d[0] * d[1] * d[2] - 1)
EOF
)
fi

get_available_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -v threshold="$MEM_THRESHOLD" -F', ' '$2 < threshold { print $1; exit }'
}

wait_for_gpu() {
  # Prints a free GPU id, blocking (with retry) until one shows up.
  local gpu_id
  while true; do
    gpu_id=$(get_available_gpu)
    if [[ -n "$gpu_id" ]]; then
      echo "$gpu_id"
      return
    fi
    echo "No GPU available. Retrying in ${SLEEP_BETWEEN}s..." >&2
    sleep "$SLEEP_BETWEEN"
  done
}

echo "=========================================================="
echo "BigCity aerial pipeline"
echo "  COARSE_CONFIG = config/${COARSE_CONFIG}.yaml"
echo "  CONFIG        = config/${CONFIG}.yaml"
echo "  MAX_BLOCK_ID  = ${MAX_BLOCK_ID}"
echo "=========================================================="

# ----- Stage 1: coarse global Gaussian model -----------------------------
if [[ "${SKIP_COARSE:-0}" != "1" ]]; then
  gpu_id=$(wait_for_gpu)
  echo ""
  echo "[1/6] Coarse training on GPU ${gpu_id}"
  CUDA_VISIBLE_DEVICES="$gpu_id" python train_large.py \
    --config "config/${COARSE_CONFIG}.yaml" 
else
  echo "[1/6] SKIP coarse (SKIP_COARSE=1)"
fi

# ----- Stage 2: data partitioning ---------------------------------------
if [[ "${SKIP_PARTITION:-0}" != "1" ]]; then
  gpu_id=$(wait_for_gpu)
  echo ""
  echo "[2/6] Data partitioning on GPU ${gpu_id}"
  CUDA_VISIBLE_DEVICES="$gpu_id" python data_partition.py \
    --config "config/${CONFIG}.yaml"
else
  echo "[2/6] SKIP partition (SKIP_PARTITION=1)"
fi

# ----- Stage 3: per-block fine-tune (parallel across free GPUs) ---------
if [[ "${SKIP_BLOCKS:-0}" != "1" ]]; then
  echo ""
  echo "[3/6] Per-block fine-tuning (blocks 0..${MAX_BLOCK_ID})"
  port="$START_PORT"
  for block_id in $(seq 0 "$MAX_BLOCK_ID"); do
    gpu_id=$(wait_for_gpu)
    echo "  -> block ${block_id} on GPU ${gpu_id} (port ${port})"
    CUDA_VISIBLE_DEVICES="$gpu_id" WANDB_MODE=offline python train_large.py \
      --config "config/${CONFIG}.yaml" \
      --block_id "$block_id" \
      --port "$port" &
    ((port++))
    sleep "$SLEEP_BETWEEN"
  done
  wait
else
  echo "[3/6] SKIP block training (SKIP_BLOCKS=1)"
fi

# ----- Stage 4: merge ---------------------------------------------------
if [[ "${SKIP_MERGE:-0}" != "1" ]]; then
  gpu_id=$(wait_for_gpu)
  echo ""
  echo "[4/6] Merge blocks on GPU ${gpu_id}"
  CUDA_VISIBLE_DEVICES="$gpu_id" python merge.py \
    --config "config/${CONFIG}.yaml"
else
  echo "[4/6] SKIP merge (SKIP_MERGE=1)"
fi

# ----- Stage 5: render the test split -----------------------------------
if [[ "${SKIP_RENDER:-0}" != "1" ]]; then
  gpu_id=$(wait_for_gpu)
  echo ""
  echo "[5/6] Rendering test split on GPU ${gpu_id}"
  # --eval loads transforms_test.json from the config's source_path;
  # --skip_train also skips parsing the ~51k train frames at startup.
  # Add --load_vq if you want to load a compressed model.
  CUDA_VISIBLE_DEVICES="$gpu_id" python render_large.py \
    --config "config/${CONFIG}.yaml" \
    --eval \
    --skip_train
else
  echo "[5/6] SKIP render (SKIP_RENDER=1)"
fi

# ----- Stage 6: metrics -------------------------------------------------
if [[ "${SKIP_METRICS:-0}" != "1" ]]; then
  gpu_id=$(wait_for_gpu)
  echo ""
  echo "[6/6] Metrics on GPU ${gpu_id}"
  CUDA_VISIBLE_DEVICES="$gpu_id" python metrics_large.py \
    --config "config/${CONFIG}.yaml" \
    -t "$out_name" \
    --use_cc
else
  echo "[6/6] SKIP metrics (SKIP_METRICS=1)"
fi

echo ""
echo "=========================================================="
echo "BigCity aerial pipeline finished."
echo "=========================================================="
