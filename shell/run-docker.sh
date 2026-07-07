#!/bin/bash
best_gpu_docker() {
    GPU_ID=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t',' -k2 -rn | head -1 | cut -d',' -f1 | tr -d ' ')
    echo "Using GPU $GPU_ID"
    docker run --gpus "device=$GPU_ID" "${@}"
}


best_gpu_docker -e XLA_PYTHON_CLIENT_MEM_FRACTION=.25 --env-file ~/1-projects/channel_ulm/.env --rm -it --shm-size=8g --volume .:/workdir --volume ~/data:/mnt/nvme/data --workdir /workdir strain  "${@}"
