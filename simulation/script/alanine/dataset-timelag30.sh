cd ../../

CUDA_VISIBLE_DEVICES=$1 python dataset-timelag.py \
    --molecule alanine \
    --temperature 300.0 \
    --data_per_traj 10000 \
    --dataset_version timelag-1n-lag10-large \
    --time_lag 10 \
    --traj_dir 1nano/c5-0 \
    --traj_dir 1nano/c5-1 \
    --traj_dir 1nano/c5-2 \
    --traj_dir 1nano/c5-3 \
    --traj_dir 1nano/c5-4 \
    --traj_dir 1nano/c7ax-0 \
    --traj_dir 1nano/c7ax-1 \
    --traj_dir 1nano/c7ax-2 \
    --traj_dir 1nano/c7ax-3 \
    --traj_dir 1nano/c7ax-4


CUDA_VISIBLE_DEVICES=$1 python dataset-timelag.py \
    --molecule alanine \
    --temperature 300.0 \
    --data_per_traj 10000 \
    --dataset_version timelag-10n-lag10-large \
    --time_lag 10 \
    --traj_dir 10nano/c5-0 \
    --traj_dir 10nano/c5-1 \
    --traj_dir 10nano/c5-2 \
    --traj_dir 10nano/c5-3 \
    --traj_dir 10nano/c5-4 \
    --traj_dir 10nano/c7ax-0 \
    --traj_dir 10nano/c7ax-1 \
    --traj_dir 10nano/c7ax-2 \
    --traj_dir 10nano/c7ax-3 \
    --traj_dir 10nano/c7ax-4


CUDA_VISIBLE_DEVICES=$1 python dataset-timelag.py \
    --molecule alanine \
    --temperature 300.0 \
    --data_per_traj 10000 \
    --dataset_version timelag-250n-lag10-large \
    --time_lag 10 \
    --traj_dir 250nano/c5-0 \
    --traj_dir 250nano/c5-1 \
    --traj_dir 250nano/c5-2 \
    --traj_dir 250nano/c5-3 \
    --traj_dir 250nano/c5-4 \
    --traj_dir 250nano/c7ax-0 \
    --traj_dir 250nano/c7ax-1 \
    --traj_dir 250nano/c7ax-2 \
    --traj_dir 250nano/c7ax-3 \
    --traj_dir 250nano/c7ax-4

