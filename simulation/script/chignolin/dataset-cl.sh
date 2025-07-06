cd ../

python build_cl_dataset.py \
    --molecule chignolin \
    --temperature 300.0 \
    --dataset_size 100000 \
    --dataset_version v1 \
    --positive_sample_augmentation 100 \
    --negative_sample_augmentation 100000 \
    --traj_dir 25-01-11/16:24 \
    --traj_dir 25-01-11/16:26