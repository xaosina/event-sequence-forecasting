#!/bin/bash

EXPERIMENT_ROOT="$1"
BOOSTER="${2:-perfect}"
RUN_NAME="${3:-booster_run}"
DEVICE="${4:-cuda:2}"

REST="${EXPERIMENT_ROOT##*log/gen-paper/}"
DATASET="${REST%%/*}"
CONFIG_FILE="${EXPERIMENT_ROOT}/seed_0/config.yaml"
CHECKPOINT_DIR="${EXPERIMENT_ROOT}/seed_0/ckpt/"

CHECKPOINT_FILE=$(ls -t "$CHECKPOINT_DIR"*.ckpt | head -n1)

RUN_NAME="${RUN_NAME}"

python main.py \
  --config_path "${CONFIG_FILE}" \
  --trainer.ckpt_resume "${CHECKPOINT_FILE}" \
  --run_name "${RUN_NAME}" \
  --runner.name GenerationEvaluator \
  --runner.run_type simple \
  --runner.params.n_runs 1 \
  --trainer.verbose True \
  --overwrite_factory "[metrics/with_detection/${DATASET},metrics/boosters/${BOOSTER}]" \
  --device "${DEVICE}" \

echo "✅ Grid search completed."