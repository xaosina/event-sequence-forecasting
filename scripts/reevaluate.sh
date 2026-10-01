#!/bin/bash

EXPERIMENT_ROOT="$1"
RUN_NAME_BASE="${2:-evaluation_run/-}"
DEVICE="${3:-cuda:0}"

DATASET="${EXPERIMENT_ROOT#log/gen-paper/}"
DATASET="${DATASET%%/*}"
CONFIG_FILE="${EXPERIMENT_ROOT}/seed_0/config.yaml"
CHECKPOINT_DIR="${EXPERIMENT_ROOT}/seed_0/ckpt/"

CHECKPOINT_FILE=$(ls -t "$CHECKPOINT_DIR"*.ckpt 2>/dev/null | head -n1)

if [ -z "$CHECKPOINT_FILE" ]; then
  echo "❌ No checkpoint found in $CHECKPOINT_DIR. Stopping script."
  exit 1
fi


RUN_NAME="${RUN_NAME_BASE}"
echo "🚀 Running: $RUN_NAME"

python main.py \
  --config_path "${CONFIG_FILE}" \
  --trainer.ckpt_resume "${CHECKPOINT_FILE}" \
  --run_name "${RUN_NAME}" \
  --runner.name GenerationEvaluator \
  --runner.run_type simple \
  --runner.params.n_runs 1 \
  --trainer.verbose True \
  --device "${DEVICE}" \
  # --overwrite_factory "[metrics/with_detection/${DATASET}]" \
