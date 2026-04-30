#!/bin/bash

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

vllm_dir=$1
listen_port=$2
api_port=$3
model=$4
device=$5

source ${vllm_dir}/.venv/bin/activate
export LD_LIBRARY_PATH=""

if [ "$device" == "gpu" ]; then
    stdbuf -oL -eL python -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --port "$api_port" &
else
    stdbuf -oL -eL python -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --port "$api_port" \
        --device cpu &
fi

SERVER_PID=$!
echo "SERVER_PID=$SERVER_PID"
