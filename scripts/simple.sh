#!/bin/bash

n_gpus=1

path=${1}
device=${2:-'cuda:0'}
lr=${3:-"3.e-4"}

if [ ! -e "$path" ]; then
    echo "Error: The path '$path' does not exist."
    exit 1
fi

run_name=$(echo "$path" | sed 's|^scripts/experiments/[^/]*/||; s|\.yaml$||')

# Use an array instead of a string to handle quoting correctly
extra_args=()
if [ -n "$lr" ]; then
    extra_args+=("--optimizer.params" "{'lr': ${lr}}")
    run_name="${run_name}/lr/${lr}"
fi

python main.py \
    --config_path "$path" \
    --device "$device" \
    --run_name "$run_name" \
    "${extra_args[@]}"