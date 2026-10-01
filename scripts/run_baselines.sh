#!/bin/bash

set -e

DATASET=${1:-'age'}
device=${2:-'cuda:0'}
MODELS=("gt" "mode" "repeat" "hist_sampler")
HORIZONS=("64" "4" "16" "32")

for HORIZON in "${HORIZONS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        EXP_DIR="log/gen-paper/${DATASET}/horizon_${HORIZON}/baselines/${MODEL}"
        if [ -d "$EXP_DIR" ]; then
            echo "skip existing $EXP_DIR"
            continue
        fi

        echo "Running model: ${MODEL} | horizon: ${HORIZON}"

        python main.py \
            --config_factory "[start,datasets/${DATASET}/${DATASET},metrics/with_detection/${DATASET},horizon/${HORIZON},methods/baselines/${MODEL}]" \
            --run_name "horizon_${HORIZON}/baselines/${MODEL}"

        SRC="${EXP_DIR}/results.csv"
        DST="${EXP_DIR}/final.csv"
        META="${EXP_DIR}/meta.yaml"

        cp "${SRC}" "${DST}"

        DATE=$(TZ=Europe/Moscow date '+%Y-%m-%d %H:%M:%S')

        cat > "${META}" <<EOF
path: "${EXP_DIR}/"
date: "${DATE}"
final: true
EOF
    done
done