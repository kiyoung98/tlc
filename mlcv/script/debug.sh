cd ../

model_list=(deeplda deeptda deeptica tae vde)

for i in "${!model_list[@]}"; do
    echo "Running ${model_list[$i]}"

    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name ${model_list[$i]}  \
        hydra.run.dir=outputs/_debug \
        ++model.logger.tags=['debug'] \
        ++model.checkpoint_name=debug \
        ++steeredmd.sample_num=8 \
        ++steeredmd.simulation.k=501 \
        ++steeredmd.simulation.time_horizon=1000 \
        ++steeredmd.repeat=1
    sleep 1
done

# for model in tbgcv tbgcv-xyz; do
#     echo "Running ${model}"

#     CUDA_VISIBLE_DEVICES=$1 python main.py \
#         --config-name ${model}  \
#         hydra.run.dir=outputs/_debug \
#         ++model.logger.tags=['debug'] \
#         ++model.checkpoint_name=debug \
#         ++steeredmd.sample_num=8 \
#         ++steeredmd.simulation.k=501 \
#         ++steeredmd.simulation.time_horizon=1000 \
#         ++steeredmd.repeat=1
#     sleep 1
# done