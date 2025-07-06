cd ../


# python build-dataset-gnncv.py \
#     --molecule alanine \
#     --temperature 300.0 \
#     --dataset_size 10000 \
#     --dataset_version graph-1n-v1 \
#     --traj_dir 24-12-26/14:53 \
#     --traj_dir 24-12-26/15:08

python build-dataset-gnncv.py \
    --molecule alanine \
    --temperature 300.0 \
    --dataset_size 10000 \
    --dataset_version graph-10n-v1 \
    --traj_dir 24-12-26/15:24 \
    --traj_dir 24-12-26/17:52 

# python build-dataset-gnncv.py \
#     --molecule alanine \
#     --temperature 300.0 \
#     --dataset_size 10000 \
#     --dataset_version graph-250n-v1 \
#     --traj_dir 25-01-29/15:29 \
#     --traj_dir 25-01-29/15:30
