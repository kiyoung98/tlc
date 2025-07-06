#!/usr/bin/env bash
set -euo pipefail

cd ../
gpus=($1 $2)
models=(deeplda deeptda deeptica tae vde)

# helper: returns 0 if GPU $1 has no running compute jobs
gpu_free() {
    local gpu_id=$1
    local pids
    pids=$(nvidia-smi -i ${gpu_id} --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
    [[ -z "$pids" ]]
}

for m in "${models[@]}"; do
    echo "Waiting for a free GPU to launch $m..."
    while true; do
        for gpu in "${gpus[@]}"; do
        if gpu_free "$gpu"; then
            FREE_GPU=$gpu
            break 2
        fi
        done
        sleep 1
    done

    echo "Launching $m on GPU $FREE_GPU!"
    # CUDA_VISIBLE_DEVICES=$FREE_GPU python main.py --config-name "$m" &
    bash ./$m.sh $FREE_GPU &
    pid=$!

    # wait until that pid actually shows up on the GPU
    until nvidia-smi -i $FREE_GPU \
        --query-compute-apps=pid --format=csv,noheader | grep -q $pid; do
        sleep 0.2
    done
done

wait
echo "Finished!"