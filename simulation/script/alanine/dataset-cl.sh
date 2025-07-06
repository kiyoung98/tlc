cd ../

# 10ns 2 trajectories each
python build_cl_dataset.py \
    --molecule alanine \
    --temperature 300.0 \
    --dataset_size 100000 \
    --dataset_version cl-10n-v3 \
    --positive_sample_augmentation 10 \
    --negative_sample_augmentation 1000 \
    --traj_dir 24-12-26/15:24 \
    --traj_dir 24-12-26/17:52 