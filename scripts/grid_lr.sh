#!/bin/bash

# e.g. scripts/experiments/age/horizon_64/multitoken.yaml
job_name=${1:?Pass experiment yaml path}
device=${2:-'cuda:0'}
lr_values=${3:-""}

path="$job_name"
rest="${path#*scripts/experiments/}"

dataset="${rest%%/*}"
run_suffix="${rest#*/}"
run_suffix="${run_suffix%.yaml}"

if [ -n "$lr_values" ]; then
    IFS=',' read -r -a lrs <<< "$lr_values"
else
    lrs=("3.e-4" "1.e-4" "1.e-5" "3.e-5" "1.e-3" "3.e-3")
fi

for lr in "${lrs[@]}"; do
    run_name="${run_suffix}/lr/${lr}"
    out_dir="log/gen-paper/${dataset}/${run_name}"
    if [ -d "$out_dir" ]; then
        echo "skip existing $out_dir"
        continue
    fi
    ./scripts/simple.sh "$job_name" "$device" "$lr"
done